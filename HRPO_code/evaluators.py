import math
import numpy as np
import random
from tqdm import tqdm


class SuccessiveHalvingEvaluator:
    """ Successive Halving Evaluator """
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=40, num_prompts_per_round=10,
                 samples_per_eval=5, max_threads=1):

        out_ranks = [-1] * len(prompts)
        prompt2idx = {p: i for i, p in enumerate(prompts)}

        budget = self.config['eval_budget']
        n = len(prompts)
        S = prompts
        scores_ = {}

        for r in range(0, math.ceil(math.log2(n))):
            t_r = math.floor(budget / (len(S) * math.ceil(math.log2(n))))

            sample = random.sample(exs, min(len(exs), t_r))

            scores = scorer(self.device, predictor, S, sample)

            average = np.mean(scores)
            for score, prompt in zip(scores, S):
                if score > average:
                    out_ranks[prompt2idx[prompt]] = r

            [scores, S] = list(zip(*sorted(list(zip(scores, S)), reverse=True)))
            for i in range(len(S)):
                if S[i] not in scores_.keys():
                    scores_[S[i]] = [scores[i]]  
                else:
                    scores_[S[i]].append(scores[i])

            for (score, prompt) in zip(scores, S):
                if len(S) > self.config['beam_size']:  
                    if score > average:
                        S = tuple([x for x in S if x != prompt]) 
        prompt2scores = {}
        for i in range(len(S)):
            prompt2scores[S[i]] = sum(scores_[S[i]]) / len(scores_[S[i]])

        return prompt2scores


class SuccessiveRejectsEvaluator:
    """ Successive Rejects Evaluator """
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=40, num_prompts_per_round=10,
                 samples_per_eval=5, max_threads=1):
        assert self.config['evaluator'] in {'sr', 's-sr'}, f'unk evaluator: {self.config["evaluator"]}'

        out_ranks = [-1] * len(prompts)
        idx2prompt = {i: p for i, p in enumerate(prompts)}

        # only run the algo until the beam is full
        num_rounds = len(prompts) - self.config['beam_size'] 

        if self.config['evaluator'] == 's-sr':
            # calculate the number of datapoints to use per rejection test
            samples_per_round = math.ceil(self.config['eval_budget'] / (num_rounds * num_prompts_per_round))  
            print('samples_per_round', samples_per_round)
            if samples_per_round == 0:
                raise Exception(f"not enough budget for s-sr!budget: {self.config['eval_budget']}")

        elif self.config['evaluator'] == 'sr':
            K = len(prompts) - self.config['beam_size'] 
            log_bar_K = 0.5 + sum([1.0/i for i in range(2, K+1)]) 
            n_prev_k = 0

        scores_ = {}
        ri = 1
        with tqdm(total=len(idx2prompt), desc='sr') as pbar:
            while True:
                if len(idx2prompt) <= self.config['beam_size']: 
                    break

                if self.config['evaluator'] == 's-sr':
                    selected_data = random.sample(exs, min(len(exs), samples_per_round))
                    selected_idxs, selected_prompts = list(zip(*random.sample(idx2prompt.items(), min(num_prompts_per_round, len(idx2prompt)))))

                elif self.config['evaluator'] == 'sr':
                    selected_idxs, selected_prompts = list(zip(*idx2prompt.items()))
                    n_k = (1.0 / log_bar_K) * ((self.config['eval_budget'] - K) / (K + 1 - ri))
                    samples_per_round = int(n_k - n_prev_k)
                    samples_per_round = max(4, samples_per_round)
                    selected_data = random.sample(exs, min(len(exs), samples_per_round))
                    n_prev_k = n_k
                    if len(selected_data) == 0:
                        raise Exception(f'not enough budget for SR! budget: {self.config["eval_budget"]}')

                scores = scorer(self.device, predictor, selected_prompts, selected_data)

                ri += 1
                max_idx = scores.index(max(scores))  

                idxs_to_remove = [selected_idxs[max_idx]]

                for i in idxs_to_remove:
                    print("delllllllll:", idx2prompt[i])
                    del idx2prompt[i]  # reject the selected arm
                    out_ranks[i] = ri  # higher score is better so increase as survives
                for i in idx2prompt.keys():
                    remain_idx = selected_idxs.index(i) 
                    if selected_prompts[remain_idx] not in scores_.keys():
                        scores_[selected_prompts[remain_idx]] = [scores[remain_idx]]  
                    else:
                        scores_[selected_prompts[remain_idx]].append(scores[remain_idx])  

                pbar.update(1)

        # fill in the beam with default values
        ri += 1
        for i in range(len(out_ranks)):
            if out_ranks[i] == -1:
                out_ranks[i] = ri
        prompt2scores = {}
        for i in idx2prompt.keys():
            prompt2scores[idx2prompt[i]] = sum(scores_[idx2prompt[i]]) / len(scores_[idx2prompt[i]]) 

        return prompt2scores



class UCBBandits:
    """ Upper Confidence Bound Bandits """
    def __init__(self, num_prompts, num_samples=5, c=1.0, mode='ucb'):
        self.c = c
        assert mode in {'ucb', 'ucb-e'}
        self.mode = mode
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.reset()

    def update(self, chosen, scores):
        for i, score in zip(chosen, scores):
            self.counts[i] += self.num_samples
            self.scores[i] += score * self.num_samples

    def reset(self):
        self.counts = np.zeros(self.num_prompts)
        self.scores = np.zeros(self.num_prompts)

    def get_scores(self):
        return np.divide(self.scores, self.counts, out=np.zeros_like(self.scores), where=self.counts != 0)

    def choose(self, n, t):
        if np.sum(self.counts) == 0:  
            return random.sample(range(self.num_prompts), n) 
        scores = self.get_scores()
        counts = self.counts + 1e-3
        if self.mode == 'ucb':
            ucb_scores = scores + self.c * np.sqrt(np.log(t) / counts)
        elif self.mode == 'ucb-e':
            ucb_scores = scores + self.c * np.sqrt(self.c / counts)
        return np.argsort(ucb_scores)[:n]

    def get_infos(self):
        return self.counts


class UCBBanditEvaluator:
    """ Upper Confidence Bound Evaluator"""
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=10, num_prompts_per_round=10,
                 samples_per_eval=5, max_threads=1):
        assert self.config['evaluator'] in {'ucb', 'ucb-e'}, f'unk evaluator: {self.config["evaluator"]}'
        bandit_algo = UCBBandits(
            len(prompts), num_samples=samples_per_eval,
            mode=self.config['evaluator'],
            c=self.config['c']
        )
        
        def data_sampler(l):
            return random.sample(l, samples_per_eval)

        num_prompts_per_round = min(num_prompts_per_round, len(prompts))
        for ri in tqdm(range(rounds), desc=f'Evaluating {len(prompts)} prompts'):
            # Sample the prompts
            sampled_prompts_idx = bandit_algo.choose(num_prompts_per_round, ri)
            sampled_prompts = [prompts[i] for i in sampled_prompts_idx]  
            sampled_data = data_sampler(exs)  

            scores = scorer(self.device, predictor, sampled_prompts, sampled_data)
            bandit_algo.update(sampled_prompts_idx, scores)
            
        return bandit_algo.get_scores().tolist()


class BruteForceEvaluator:
    """ Brute Force Evaluator """
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def __call__(self, prompts, error_exs, task, predictor, scorer, rounds=40,
                 num_prompts_per_round=10, c=2.0, samples_per_eval=5, max_threads=1):
        sample_size = min(len(error_exs), int(self.config['eval_budget'] / len(prompts)))
        eval_exs = random.sample(error_exs, sample_size)

        scores = scorer(self.device, predictor, prompts, eval_exs, max_threads=max_threads)
        return scores
