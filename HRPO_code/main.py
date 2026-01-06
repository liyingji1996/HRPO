import os
import io
import json
import torch
from tqdm import tqdm
import time
from PIL import Image
from transformers import (AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration,
                          AutoModelForCausalLM, MllamaForConditionalGeneration,Phi4MultimodalVisionModel)
from transformers import BitsAndBytesConfig
import pyarrow.parquet as pq
from qwen_vl_utils import process_vision_info
import argparse
import task
import evaluators
import scorers
import predictors
import optimizers
import random
import numpy as np
import csv
from accelerate import infer_auto_device_map

def get_task_class(task_name):
    if task_name == 'SB':
        return task.SBTernaryTask
    elif task_name == 'VLBias':
        return task.VLTernaryTask
    elif task_name == 'VLBias_open':
        return task.VLOpenedTask
    else:
        raise Exception(f'Unsupported task: {task_name}')

def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')

def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'll':
        return scorers.CachedLogLikelihoodScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')

def load_model(LLM, model_path):
    print(f"Loading {LLM} from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True,
                                              padding_side='left')
    if 'qwen' in LLM.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        )
    elif 'llava' in LLM.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        )
    elif 'internvl' in LLM.lower():
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        )
    elif 'llama3.2-11b' in LLM.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  
                    )
        )
    elif 'gpt' in LLM.lower() or 'gemini' in args.LLM.lower():
        model, processor = None, None

    return processor, model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='SB', type=str)
    parser.add_argument('--LLM', default='Qwen2.5-VL-7B-Instruct', type=str)
    parser.add_argument('--provider', default='openai', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--data_dir', default='data/SB-Bench-datasets', type=str)
    parser.add_argument('--out_file', default='out_file', type=str)
    parser.add_argument('--prompts', default='prompts/bbq.md', type=str)
    parser.add_argument('--bias_type', default='Gender', type=str)
    parser.add_argument('--temperature', default=0.8, type=float)
    parser.add_argument('--rounds', default=8, type=int)
    parser.add_argument('--beam_size', default=4, type=int)  
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--minibatch_size', default=64, type=int,
                        help="The number of training samples sampled during the expansion")
    parser.add_argument('--n_gradients', default=3, type=int, help="the number of reasoning")
    parser.add_argument('--errors_per_gradient', default=4, type=int, help="Number of errors samples per gradient")
    parser.add_argument('--gradients_per_error', default=1, type=int, help="Number of gradient per errors sample")
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=4, type=int) 

    parser.add_argument('--evaluator', default="sh", type=str, help="'bf', 'ucb', 'ucb-e', 'sr', 's-sr', 'sh'")
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=4, type=int)
    parser.add_argument('--eval_prompts_per_round', default=16, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', type=bool, default=True)
    # baselines
    parser.add_argument('--pa', type=bool, default=False)
    parser.add_argument('--sd', type=bool, default=False)
    parser.add_argument('--baselines', type=bool, default=False)
    parser.add_argument('--prompts_sd', default='prompts/bbq.md', type=str)
    # ablation
    parser.add_argument('--no_history', type=bool, default=False)
    parser.add_argument('--no_ref', type=bool, default=False)
    parser.add_argument('--no_bias', type=bool, default=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = get_args()
    config = vars(args)
    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = get_task_class(args.task)(config, args.data_dir)

    processor, model = load_model(args.LLM, args.model_path)
    scorer = get_scorer(args.scorer)(config, model, processor, device)
    evaluator = get_evaluator(args.evaluator)(config, device)
    bf_eval = get_evaluator('bf')(config, device)

    if args.task == 'SB':
        gpt4 = predictors.BBQTernaryPredictor(config, model, processor)
    elif args.task == 'VLBias':
        gpt4 = predictors.VLBiasTernaryPredictor(config, model, processor)
    elif args.task == 'VLBias_open':
        gpt4 = predictors.VLBiasOpenTask(config, model, processor)

    train_exs = task.get_train_examples()
    dev_exs = task.get_dev_examples()
    test_exs = task.get_test_examples()

    if not os.path.exists(args.out_file):
        os.makedirs(args.out_file)
    out_path = os.path.join(args.out_file, '{}_{}_{}_{}_results.txt'.format(args.LLM, args.task, args.bias_type, args.evaluator))
    with open(out_path, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = []
    if os.path.isdir(args.prompts):
        for filename in sorted(os.listdir(args.prompts)):
            filepath = os.path.join(args.prompts, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    candidates.append(f.read())
    elif os.path.isfile(args.prompts):
        with open(args.prompts, 'r', encoding='utf-8') as f:
            candidates.append(f.read())
    else:
        raise ValueError(f"Invalid path: {args.prompts}")

    if args.sd:
        candidates_sd = []
        for filename in sorted(os.listdir(args.prompts_sd)):
            filepath = os.path.join(args.prompts_sd, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    candidates_sd.append(f.read())
    else:
        candidates_sd = [None]

    optimizer = optimizers.ProTeGi(config, evaluator, scorer, bf_eval)

    min_score = 2.0  
    stop = 0  
    for round in tqdm(range(config['rounds'] + 1)):
        if stop == 2:
            break
        print("STARTING ROUND ", round)
        start = time.time()
        with open(out_path, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")

        # expand candidates
        if round > 0: 
            candidates, oldp_feedback_newp = optimizer.expand_candidates(device, model, processor, candidates, task,
                                                                         gpt4, train_exs, round)
            with open(out_path, 'a', encoding='utf-8') as outf:
                outf.write(f'Optimization Candidates Prompts: {candidates}\n\n')
                for (i, item) in enumerate(oldp_feedback_newp):
                    outf.write(f'The {i}-th Group:\n')
                    outf.write(f'Old prompt: {item[0]}\n')
                    outf.write(f'Feedback: {item[1]}\n')
                    outf.write(f'New prompt: {item[2]}\n\n')
                    outf.write(f'Good History: {item[3]}\n\n')
                    outf.write(f'Bad History: {item[4]}\n\n')
                    outf.write(f'Examples: {item[5]}\n\n')
                    outf.write(f'IDs: {item[6]}\n\n')

        # score candidates
        if args.evaluator not in ['sr', 's-sr', 'sh'] or round == 0:
            scores = optimizer.score_candidates(candidates, task, gpt4, dev_exs, round)
            print("score candidates", scores)
        else:
            prompt2scores = optimizer.score_candidates(candidates, task, gpt4, dev_exs, round)
            print("prompt2scores:", prompt2scores)
            candidates = list(prompt2scores.keys())
            scores = list(prompt2scores.values())
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates))))) 
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]

        if round == 0:
            min_score = 1.0
        elif min_score > scores[0]:
            min_score = scores[0]
        else:
            stop += 1
            print("stop number:", stop)

        with open(out_path, 'a') as outf:
            outf.write(f'Final Prompts: {candidates}\n\n')
            outf.write(f'Dev Scores: {scores}\n\n')
        metrics = []

        candidate2scores = []
        for candidate, score in zip(candidates, scores):
            if args.task == 'SB' or args.task == 'VLBias':
                if 'gpt' in args.LLM.lower() or 'gemini' in args.LLM.lower():
                    score_, _, _, _, _, _, _, _, _, _, _ = task.bbq_evaluate(device, gpt4, candidate, test_exs, n=50, prompt_sd=candidates_sd[0])  
                else:
                    score_, _, _, _, _, _, _, _, _, _, _ = task.bbq_evaluate(device, gpt4, candidate, test_exs, n=len(test_exs), prompt_sd=candidates_sd[0])  
            elif args.task == 'VLBias_open':
                if 'gpt' in args.LLM.lower() or 'gemini' in args.LLM.lower():
                    score_, _, _, _, _, _, _ = task.bbq_evaluate(device, gpt4, candidate, test_exs, round, n=50, prompt_sd=candidates_sd[0])
                else:
                    score_, _, _, _, _, _, _ = task.bbq_evaluate(device, gpt4, candidate, test_exs, round, n=len(test_exs), prompt_sd=candidates_sd[0]) 
            print("Test Bias Scores:", score_)
            metrics.append(score_)
            candidate2scores.append([score_, candidate])

        with open(out_path, 'a') as outf:
            outf.write(f'Test Scores: {metrics}\n\n')
            outf.write(f'min_score: {min_score}\n\n')
            outf.write(f'Time: {time.time() - start}s\n\n\n\n')

        csv_file = os.path.join(args.out_file, '{}_{}_{}.csv'.format(args.task, args.LLM, args.bias_type))
        if args.baselines:
            csv_header = ["Task", "LLM", "bias_type", "round", "Score_1", "Score_2", "Score_3", "Prompt_Scores"]
        else:
            csv_header = ["Task", "LLM", "bias_type", "round", "Prompt_Scores"]

        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)

        if args.baselines:
            candidate2scores_ = np.mean([cascore[0] for cascore in candidate2scores])
            row = [args.task, args.LLM, args.bias_type, round, candidate2scores[0][0], candidate2scores[1][0], candidate2scores[2][0], candidate2scores_]
        else:
            row = [args.task, args.LLM, args.bias_type, round, candidate2scores]

        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        for i in range(len(metrics)):
            if metrics[i] == '0.0':
                break

    print("DONE!")