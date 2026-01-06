import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils
import math
from itertools import combinations
import statistics
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, bf_eval=None):
        self.args = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.bf_eval = bf_eval
        self.prompt_histories = {}

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass

class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """
    def bbq_sample_error_str(self, images, texts, labels, ids, preds, task, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p: 
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_ids = [ids[i] for i in sample_idxs]
        sample_texts = [texts[i] for i in sample_idxs]
        sample_images = [images[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = []
        for i in sample_idxs:
            sample_preds.append(preds[i])
        error_string = ''
        error_images, error_ids = [], []
        error_idx = 0
        for i, (d, m, t, l, p) in enumerate(zip(sample_ids, sample_images, sample_texts, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx+1}\n'
            error_string += f'Text: \"{t.strip()}\"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\n\n'
            error_ids.append(d)
            error_images.append(m)
            error_idx += 1
        return error_string.strip(), error_images, error_ids

    def opentask_sample_error_str(self, images, texts, scores, ids, answers, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, score in enumerate(scores):
            if score >= 0.5 or score <= -0.3: 
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_ids = [ids[i] for i in sample_idxs]
        sample_texts = [texts[i] for i in sample_idxs]
        sample_images = [images[i] for i in sample_idxs]
        sample_labels = [scores[i] for i in sample_idxs]
        sample_preds = []
        for i in sample_idxs:
            sample_preds.append(answers[i])
        error_string = ''
        error_images, error_ids = [], []
        error_idx = 0
        for i, (d, m, t, l, p) in enumerate(zip(sample_ids, sample_images, sample_texts, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx+1}\n'
            error_string += f'Task: \"{t.strip()}\"\nResponse: {p}\nScore: {l}\n\n'
            error_ids.append(d)
            error_images.append(m)
            error_idx += 1
        return error_string.strip(), error_images, error_ids

    def parse_tagged_text(self, text, start_tag, end_tag_1):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag_1, start_index)
            s = 0
            if end_index == -1:
                s = 1
                end_index = text.find("</END>", start_index)
                # end_index = text.find("</START>", start_index)
                if end_index == -1:
                    break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            if s == 1:
                text = text[end_index+len("</END>"):]
            elif s == 0:
                text = text[end_index + len(end_tag_1):]
        return texts

    def print_prompt_history_new(self, history_dict, current_prompt, score):
        history_chain = []
        prompt = current_prompt
        visited = set()

        while prompt in history_dict:
            if prompt in visited:
                break
            visited.add(prompt)

            parent_prompt, parent_score = history_dict[prompt]
            if parent_prompt is not None:
                history_chain.append((parent_prompt, parent_score))
            prompt = parent_prompt

        if not history_chain:
            return "", ""

        history_chain.reverse()

        history_str_good, history_str_bad = [], []
        for idx, (text, sc) in enumerate(history_chain, start=1):
            entry = f'Prompt of the {idx}-th iteration: "{text}" (Bias Score: {sc})'
            if sc < score:
                history_str_good.append(entry)
            else:
                history_str_bad.append(entry)
        history_str_good = "\n".join(history_str_good)
        history_str_bad = "\n".join(history_str_bad)

        return history_str_good, history_str_bad

    def _get_gradients(self, device, model, tokenizer, prompt, score, fair_score, iteration, history_str_good, history_str_bad, error_images, error_string,  n=1):
        if self.args['bias_type'] == 'Race_x_gender':
            gradient_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Task]
                    Using the information above, analyze the strengths of the good prompts and the weaknesses of the bad ones. Provide one clear and actionable suggestion to improve the current prompt, aiming to reduce gender bias and race bias and approach the target bias score of {fair_score}.

                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        elif self.args['bias_type'] == 'Race_x_SES':
            gradient_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Task]
                    Using the information above, analyze the strengths of the good prompts and the weaknesses of the bad ones. Provide one clear and actionable suggestion to improve the current prompt, aiming to reduce race bias and socioeconomic status bias and approach the target bias score of {fair_score}.

                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        else:
            gradient_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Task]
                    Using the information above, analyze the strengths of the good prompts and the weaknesses of the bad ones. Provide one clear and actionable suggestion to improve the current prompt, aiming to reduce {self.args['bias_type']} bias and approach the target bias score of {fair_score}.
                    
                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        batch_data = [(error_images, gradient_prompt)]
        res = utils.ds2(self.args, device, model, tokenizer, batch_data, max_tokens=1024, n=n)
        feedbacks = []
        for r in res:
            feedbacks += [r]
        return feedbacks

    def _get_gradients_no_bias(self, device, model, tokenizer, prompt, score, fair_score, iteration, history_str_good, history_str_bad, n=1):
        if self.args['bias_type'] == 'gender_race':
            gradient_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.

                    [Good Prompt History]
                    {history_str_good}

                    [Bad Prompt History]
                    {history_str_bad}

                    [Task]
                    Using the information above, analyze the strengths of the good prompts and the weaknesses of the bad ones. Provide one clear and actionable suggestion to improve the current prompt, aiming to reduce gender bias and race bias and approach the target bias score of {fair_score}.
                    
                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        elif self.args['bias_type'] == 'race_ses':
            gradient_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.

                    [Good Prompt History]
                    {history_str_good}

                    [Bad Prompt History]
                    {history_str_bad}

                    [Task]
                    Using the information above, analyze the strengths of the good prompts and the weaknesses of the bad ones. Provide one clear and actionable suggestion to improve the current prompt, aiming to reduce race bias and socioeconomic status bias and approach the target bias score of {fair_score}.
                    
                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        else:
            gradient_prompt = f"""
                            [Iteration] {iteration}.
                            [Current Prompt] "{prompt}".
                            [Bias Score] {score}.

                            [Good Prompt History]
                            {history_str_good}
        
                            [Bad Prompt History]
                            {history_str_bad}

                            [Task]
                            Using the information above, analyze the strengths of the good prompts and the weaknesses of the bad ones. Provide one clear and actionable suggestion to improve the current prompt, aiming to reduce {self.args['bias_type']} bias and approach the target bias score of {fair_score}.
                    
                            [Constraints]
                            1. Output only a natural language suggestion describing what to change and why.
                            2. Do NOT output a full rewritten prompt or direct text substitutions.
                            3. Make the suggestion specific and implementable by a human or system in the next iteration.

                            [Self-Check Before Answering]
                            If your output contains a rewritten prompt, rephrase it into a suggestion only.
                            """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        error_images = None
        batch_data = [(error_images, gradient_prompt)]
        res = utils.ds2(self.args, device, model, tokenizer, batch_data, max_tokens=1024, n=n)
        feedbacks = []
        for r in res:
            feedbacks += [r]
        return feedbacks

    def _get_gradients_no_his(self, device, model, tokenizer, prompt, score, fair_score, error_images, error_string, n=1):
        if self.args['bias_type'] == 'gender_race':
            gradient_prompt = f"""
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Task]
                    Using the information above, provide one clear and actionable suggestion to improve the current prompt, aiming to reduce gender bias and race bias and approach the target bias score of {fair_score}.

                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        elif self.args['bias_type'] == 'race_ses':
            gradient_prompt = f"""
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Task]
                    Using the information above, provide one clear and actionable suggestion to improve the current prompt, aiming to race bias and socioeconomic status bias and approach the target bias score of {fair_score}.

                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        else:
            gradient_prompt = f"""
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Task]
                    Using the information above, provide one clear and actionable suggestion to improve the current prompt, aiming to {self.args['bias_type']} bias and approach the target bias score of {fair_score}.

                    [Constraints]
                    1. Output only a natural language suggestion describing what to change and why.
                    2. Do NOT output a full rewritten prompt or direct text substitutions.
                    3. Make the suggestion specific and implementable by a human or system in the next iteration.

                    [Self-Check Before Answering]
                    If your output contains a rewritten prompt, rephrase it into a suggestion only.
                    """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        batch_data = [(error_images, gradient_prompt)]
        res = utils.ds2(self.args, device, model, tokenizer, batch_data, max_tokens=1024, n=n) 
        feedbacks = []
        for r in res:
            feedbacks += [r]
        return feedbacks

    def _apply_gradient(self, device, model, tokenizer, prompt, score, fair_score, feedback_str, history_str_good, history_str_bad, error_string, error_images, iteration, n=1):
        """ Incorporate feedback gradient into a prompt."""
        if self.args['bias_type'] == 'Race_x_gender':
            transformation_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize gender bias and race bias, targeting a bias score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        elif self.args['bias_type'] == 'Race_x_SES':
            transformation_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize race bias and socioeconomic status bias, targeting a bias score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        else:
            transformation_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize {self.args['bias_type']} bias, targeting a bias score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        batch_data = [(error_images, transformation_prompt)]
        res = utils.ds2(self.args, device, model, tokenizer, batch_data, max_tokens=1024, n=n)
        new_prompts = []
        for r in res:
            if len(new_prompts) < n:
                new_prompts += [r]
        return new_prompts

    def _apply_gradient_no_bias(self, device, model, tokenizer, prompt, score, fair_score, feedback_str, history_str_good, history_str_bad, iteration, n=1):
        """ Incorporate feedback gradient into a prompt."""
        if self.args['bias_type'] == 'gender_race':
            transformation_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize gender bias and race bias, targeting a bias score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        elif self.args['bias_type'] == 'race_ses':
            transformation_prompt = f"""
                    [Iteration] {iteration}.
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.

                    [Good Prompt History]
                    {history_str_good}
                    
                    [Bad Prompt History]
                    {history_str_bad}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize race bias and socioeconomic status bias, targeting a bias score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        else:
            transformation_prompt = f"""
                                    [Iteration] {iteration}.
                                    [Current Prompt] "{prompt}".
                                    [Bias Score] {score}.

                                    [Good Prompt History]
                                    {history_str_good}
                                    
                                    [Bad Prompt History]
                                    {history_str_bad}
                
                                    [Improvement Suggestion] "{feedback_str}".

                                    [Task]
                                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize {self.args['bias_type']} bias, targeting a bias score of {fair_score}.

                                    [Constraints]
                                    1. Output only the optimized prompt text.
                                    2. Do NOT include any explanations, commentary, or unrelated content.
                                    3. The output should be a ready-to-use prompt for the next iteration.
                                    """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        error_images = None
        batch_data = [(error_images, transformation_prompt)]
        res = utils.ds2(self.args, device, model, tokenizer, batch_data, max_tokens=1024, n=n)
        new_prompts = []
        for r in res:
            if len(new_prompts) < n:
                new_prompts += [r]
        return new_prompts

    def _apply_gradient_no_ref(self, device, model, tokenizer, prompt, score, fair_score, history_str_good, history_str_bad, error_string, error_images, iteration, n=1):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""[Iteration] {iteration}.
                                    [Current Prompt] "{prompt}".
                                    [Bias Score] {score}.
                                    
                                    [Biased Prediction Examples]
                                    {error_string}
                
                                    [Good Prompt History]
                                    {history_str_good}
                
                                    [Bad Prompt History]
                                    {history_str_bad}

                                    [Task]
                                    Based on the above information, optimize the current prompt according to minimize {self.args['bias_type']} bias, targeting a bias score of {fair_score}.

                                    [Constraints]
                                    1. Output only the optimized prompt text.
                                    2. Do NOT include any explanations, commentary, or unrelated content.
                                    3. The output should be a ready-to-use prompt for the next iteration.
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        batch_data = [(error_images, transformation_prompt)]
        res = utils.ds2(self.args, device, model, tokenizer, batch_data, max_tokens=1024, n=n)
        new_prompts = []
        for r in res:
            if len(new_prompts) < n:
                new_prompts += [r]
        return new_prompts

    def _apply_gradient_no_his(self, device, model, tokenizer, prompt, score, fair_score, feedback_str, error_string, error_images, n=1):
        if self.args['bias_type'] == 'gender_race':
            transformation_prompt = f"""
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize gender bias and race bias, targeting a bias score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        elif self.args['bias_type'] == 'race_ses':
            transformation_prompt = f"""
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize race bias and socioeconomic status bias, targeting a bias score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        else:
            transformation_prompt = f"""
                    [Current Prompt] "{prompt}".
                    [Bias Score] {score}.
                                    
                    [Biased Prediction Examples]
                    {error_string}

                    [Improvement Suggestion] "{feedback_str}".

                    [Task]
                    Based on the above information, optimize the current prompt according to the improvement suggestion to minimize {self.args['bias_type']} bias, targeting a fairness score of {fair_score}.

                    [Constraints]
                    1. Output only the optimized prompt text.
                    2. Do NOT include any explanations, commentary, or unrelated content.
                    3. The output should be a ready-to-use prompt for the next iteration.
                    """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        batch_data = [(error_images, transformation_prompt)]
        res = utils.ds2(self.args, device, model, tokenizer, batch_data, max_tokens=1024, n=n)
        new_prompts = []
        for r in res:
            if len(new_prompts) < n:
                new_prompts += [r]
        return new_prompts

    def bbq_get_gradients(self, device, model, tokenizer, task_section, task, images, texts, labels, preds, score, fair_score, iteration, history, ids):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        history_str_good, history_str_bad = self.print_prompt_history_new(history, task_section, score)
        print("history_str_good", history_str_good)
        print("history_str_bad", history_str_bad)
        for _ in tqdm(range(self.args['n_gradients']), total=self.args['n_gradients'], desc='gradients..'):
            error_string, error_images, error_ids = self.bbq_sample_error_str(images, texts, labels, ids, preds, task,
                                                     n=self.args['errors_per_gradient'])  
            gradients = self._get_gradients(device, model, tokenizer, task_section, score, fair_score, iteration, history_str_good, history_str_bad, error_images, error_string, n=1)  # 为每组4个error样本返回一个gradient
            print("gradients", gradients)
            prompt_feedbacks += [(t, history_str_good, history_str_bad, error_string, error_images, error_ids) for t in gradients]
        print("prompt_feedbacks", prompt_feedbacks)
        return prompt_feedbacks

    def opentask_get_gradients(self, device, model, tokenizer, task_section, images, texts, socres, preds, score, fair_score, iteration, history, ids):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        history_str_good, history_str_bad = self.print_prompt_history_new(history, task_section, score)
        print("history_str_good", history_str_good)
        print("history_str_bad", history_str_bad)
        for _ in tqdm(range(self.args['n_gradients']), total=self.args['n_gradients'], desc='gradients..'):
            error_string, error_images, error_ids = self.opentask_sample_error_str(images, texts, socres, ids, preds, n=self.args['errors_per_gradient'])  # 返回采样的4条错误样本，字符串拼接在一起
            gradients = self._get_gradients(device, model, tokenizer, task_section, score, fair_score, iteration, history_str_good, history_str_bad, error_images, error_string, n=1)  # 为每组4个error样本返回一个gradient
            print("gradients", gradients)
            prompt_feedbacks += [(t, history_str_good, history_str_bad, error_string, error_images, error_ids) for t in gradients]
        print("prompt_feedbacks", prompt_feedbacks)
        return prompt_feedbacks

    def bbq_get_gradients_no_ref(self, task_section, task, images, texts, labels, preds, score, history, ids):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        history_str_good, history_str_bad = self.print_prompt_history_new(history, task_section, score)
        print("history_str_good", history_str_good)
        print("history_str_bad", history_str_bad)
        for _ in tqdm(range(self.args['n_gradients']), total=self.args['n_gradients'], desc='gradients..'):
            error_string, error_images, error_ids = self.bbq_sample_error_str(images, texts, labels, ids, preds, task,
                                                                              n=self.args['errors_per_gradient'])
            prompt_feedbacks += [(history_str_good, history_str_bad, error_string, error_images, error_ids)]
        print("prompt_feedbacks", prompt_feedbacks)
        return prompt_feedbacks

    def bbq_get_gradients_no_his(self, device, model, tokenizer, task_section, task, images, texts, labels, preds, score, fair_score, ids):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.args['n_gradients']), total=self.args['n_gradients'], desc='gradients..'):
            error_string, error_images, error_ids = self.bbq_sample_error_str(images, texts, labels, ids, preds, task, n=self.args['errors_per_gradient'])
            gradients = self._get_gradients_no_his(device, model, tokenizer, task_section, score, fair_score, error_images, error_string, n=1)
            print("gradients", gradients)
            prompt_feedbacks += [(t, error_string, error_images, error_ids) for t in gradients]
        print("prompt_feedbacks", prompt_feedbacks)
        return prompt_feedbacks

    def bbq_get_gradients_no_bias(self, device, model, tokenizer, task_section, score, fair_score, iteration, history):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        history_str_good, history_str_bad = self.print_prompt_history_new(history, task_section, score)
        print("history_str_good", history_str_good)
        print("history_str_bad", history_str_bad)
        for _ in tqdm(range(self.args['n_gradients']), total=self.args['n_gradients'], desc='gradients..'):
            gradients = self._get_gradients_no_bias(device, model, tokenizer, task_section, score, fair_score, iteration, history_str_good, history_str_bad, n=1)  # 为每组4个error样本返回一个gradient
            print("gradients", gradients)
            prompt_feedbacks += [(t, history_str_good, history_str_bad) for t in gradients]
        print("prompt_feedbacks", prompt_feedbacks)
        return prompt_feedbacks

    def expand_candidates(self, device, model, tokenizer, prompts, task, gpt4, train_exs, iteration):
        if iteration == 1:
            for prompt in prompts:
                sections_ = utils.parse_sectioned_prompt(prompt)
                task_section_ = sections_['task'].strip()
                self.prompt_histories[task_section_] = (None, None)

        minibatch = random.sample(train_exs, k=self.args['minibatch_size'])

        new_prompts = []
        oldp_feedback_newp = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = utils.parse_sectioned_prompt(prompt)  
            task_section = sections['task'].strip()

            # evaluate prompt on minibatch
            if self.args['task'] == 'SB' or self.args['task'] == 'VLBias':
                (score, images, texts, labels, preds, stereo_groups, ans0_infos, ans1_infos, ans2_infos, preds_all, ids) \
                    = task.bbq_evaluate(device, gpt4, prompt, minibatch, n=self.args['minibatch_size'])  

                # get gradients
                new_task_sections = []
                fair_score = 0
                if self.args['n_gradients'] > 0:
                    if not self.args['no_bias'] and not self.args['no_history'] and not self.args['no_ref']:
                        gradients = self.bbq_get_gradients(device, model, tokenizer, task_section, task, images, texts, labels, preds,
                                                            score, fair_score, iteration, self.prompt_histories, ids) 
                        for feedback, history_str_good, history_str_bad, error_string, error_images, error_ids in tqdm(gradients, desc='applying gradients'):
                            tmp = self._apply_gradient(device, model, tokenizer, task_section, score, fair_score, feedback, history_str_good, history_str_bad, error_string, error_images, iteration)  # 为每组（gradient, history）新生成一个prompt，
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score) 
                            new_task_sections += tmp
                            oldp_feedback_newp.append([task_section, feedback, tmp, history_str_good, history_str_bad, error_string, error_ids])
                        print("new_task_sections", new_task_sections)
                    elif self.args['no_ref']:
                        gradients = self.bbq_get_gradients_no_ref(task_section, task, images, texts, labels, preds, score, self.prompt_histories, ids)
                        for history_str_good, history_str_bad, error_string, error_images, error_ids in tqdm(gradients, desc='applying gradients'):
                            tmp = self._apply_gradient_no_ref(device, model, tokenizer, task_section, score, fair_score, history_str_good, history_str_bad, error_string, error_images, iteration)
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score) 
                            new_task_sections += tmp
                            feedback = None
                            oldp_feedback_newp.append([task_section, feedback, tmp, history_str_good, history_str_bad, error_string, error_ids])
                        print("new_task_sections", new_task_sections)
                    elif self.args['no_bias']:
                        gradients = self.bbq_get_gradients_no_bias(device, model, tokenizer, task_section, score, fair_score, iteration, self.prompt_histories)  # 返回一个list，包含4组（gradient, 4个errors）
                        for feedback, history_str_good, history_str_bad in tqdm(gradients, desc='applying gradients'):
                            tmp = self._apply_gradient_no_bias(device, model, tokenizer, task_section, score, fair_score, feedback, history_str_good, history_str_bad, iteration)  # 为每组（gradient, history）新生成一个prompt，
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score) 
                            new_task_sections += tmp
                            error_string, error_ids = None, None
                            oldp_feedback_newp.append([task_section, feedback, tmp, history_str_good, history_str_bad, error_string, error_ids])
                        print("new_task_sections", new_task_sections)
                    elif self.args['no_history']:
                        gradients = self.bbq_get_gradients_no_his(device, model, tokenizer, task_section, task, images, texts, labels, preds, score, fair_score, ids)
                        for feedback, error_string, error_images, error_ids in tqdm(gradients, desc='applying gradients'):
                            tmp = self._apply_gradient_no_his(device, model, tokenizer, task_section, score, fair_score, feedback, error_string, error_images)  # 为每组（gradient, history）新生成一个prompt，
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score)
                            new_task_sections += tmp
                            history_str_good, history_str_bad = None, None
                            oldp_feedback_newp.append([task_section, feedback, tmp, history_str_good, history_str_bad, error_string, error_ids])
                        print("new_task_sections", new_task_sections)

                # combine
                new_sections = new_task_sections
                new_sections = list(set(new_sections))  # dedup
                tmp_new_prompts = [prompt.replace(task_section, tmp) for tmp in new_sections]

                # filter a little
                if len(new_sections) > self.args['max_expansion_factor']:
                    if self.args['reject_on_errors']:
                        error_exs = []
                        for i, (t, m, l, p, s, a0, a1, a2) in enumerate(zip(texts, images, labels, preds, stereo_groups, ans0_infos, ans1_infos, ans2_infos)):
                            if l != p:
                                error_exs.append({'text': t, 'image': m, 'label': l, 'stereo_group': s, 'ans0_info': a0, 'ans1_info': a1, 'ans2_info': a2})

                        error_exs = random.sample(error_exs, min(len(error_exs), 32))

                        error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer)
                        tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[:self.args['max_expansion_factor']]]
                    else:
                        tmp_new_prompts = random.sample(tmp_new_prompts, k=self.args['max_expansion_factor'])
                        # evaluate prompt on minibatch
            elif self.args['task'] == 'VLBias_open':
                score, images, texts, stereo_groups, answers, scores, ids = task.bbq_evaluate(device, gpt4, prompt, minibatch, n=self.args['minibatch_size'])

                # get gradients
                new_task_sections = []
                fair_score = 0
                if self.args['n_gradients'] > 0:
                    if not self.args['no_bias'] and not self.args['no_history'] and not self.args['no_ref']:
                        gradients = self.opentask_get_gradients(device, model, tokenizer, task_section, images, texts,
                                                                scores, answers, score, fair_score, iteration, self.prompt_histories, ids)
                        for feedback, history_str_good, history_str_bad, error_string, error_images, error_ids in tqdm(
                                gradients, desc='applying gradients'):
                            tmp = self._apply_gradient(device, model, tokenizer, task_section, score,
                                                           fair_score, feedback, history_str_good,
                                                           history_str_bad, error_string, error_images,
                                                           iteration)  
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score)  
                            new_task_sections += tmp
                            oldp_feedback_newp.append(
                                [task_section, feedback, tmp, history_str_good, history_str_bad,
                                 error_string, error_ids])

                        print("new_task_sections", new_task_sections)
                    elif self.args['no_ref']:
                        gradients = self.bbq_get_gradients_no_ref(task_section, task, texts, labels, preds,
                                                                  self.prompt_histories)
                        for history_str, error_string in tqdm(gradients, desc='applying gradients'):
                            tmp = self._apply_gradient_no_ref(device, model, tokenizer, task_section,
                                                                  score,
                                                                  fair_score, history_str, error_string,
                                                                  iteration)
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score) 
                            new_task_sections += tmp
                            feedback = None
                            oldp_feedback_newp.append(
                                [task_section, feedback, tmp, history_str, error_string, error_ids])
                        print("new_task_sections", new_task_sections)
                    elif self.args['no_bias']:
                        gradients = self.bbq_get_gradients_no_bias(device, model, tokenizer, task_section,
                                                                   score, fair_score,
                                                                   iteration,
                                                                   self.prompt_histories)  
                        for feedback, history_str in tqdm(gradients, desc='applying gradients'):
                            tmp = self._apply_gradient_no_bias(device, model, tokenizer, task_section,
                                                                   score, fair_score, feedback, history_str,
                                                                   iteration) 
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score)
                            new_task_sections += tmp
                            error_string = None
                            oldp_feedback_newp.append(
                                [task_section, feedback, tmp, history_str, error_string, error_ids])
                        print("new_task_sections", new_task_sections)
                    elif self.args['no_history']:
                        gradients = self.bbq_get_gradients_no_his(device, model, tokenizer, task_section,
                                                                  task, images, texts, labels,
                                                                  preds, score, fair_score, ids)
                        for feedback, error_string, error_images, error_ids in tqdm(gradients,
                                                                                    desc='applying gradients'):
                            tmp = self._apply_gradient_no_his(device, model, tokenizer, task_section, score,
                                                              fair_score, feedback, error_string,
                                                              error_images) 
                            for child in tmp:
                                self.prompt_histories[child] = (task_section, score)
                            new_task_sections += tmp
                            history_str = None
                            oldp_feedback_newp.append(
                                [task_section, feedback, tmp, history_str, error_string, error_ids])
                        print("new_task_sections", new_task_sections)

                # combine
                new_sections = new_task_sections
                new_sections = list(set(new_sections))  # dedup
                tmp_new_prompts = [prompt.replace(task_section, tmp) for tmp in new_sections]

                # filter a little
                if len(new_sections) > self.args['max_expansion_factor']:
                    if self.args['reject_on_errors']:
                        error_exs = []
                        for i, (t, m, l, p, s) in enumerate(zip(texts, images, scores, answers, stereo_groups)):
                            if l >= 0.5 or l <= -0.3:
                                error_exs.append({'text': t, 'image': m, 'score': l, 'answers': p,'stereo_group': s})

                        error_exs = random.sample(error_exs, min(len(error_exs), 32))
                        error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer)
                        tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[:self.args['max_expansion_factor']]]
                    else:
                        tmp_new_prompts = random.sample(tmp_new_prompts,k=self.args['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts  # add originals
        new_prompts = list(set(new_prompts))  # dedup
        print("new_prompts:", new_prompts)

        return new_prompts, oldp_feedback_newp

    def score_candidates(self, prompts, task, gpt4, train_exs, round):
        """ Score a list of prompts."""
        if len(prompts) == 1 or round == 0: 
            score = []
            for prompt in prompts:
                score.append(1.0)
            return score

        evals = self.evaluator_fn(
            prompts, train_exs, task, gpt4,
            scorer=self.scorer,
            rounds=self.args['eval_rounds'],
            num_prompts_per_round=self.args['eval_prompts_per_round'],
            samples_per_eval=self.args['samples_per_eval'],
        )
        return evals
