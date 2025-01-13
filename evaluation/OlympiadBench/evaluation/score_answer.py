from evaluate import extract_answer, MathJudger
import json
import random
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm
import numpy as np
import argparse
import re


def read_jsonl(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data 


def extract_box(latex_str):
    # 查找所有的 \boxed{...} 结构
    boxed_matches = re.finditer(r'\\boxed{', latex_str)
    results = ""

    for match in boxed_matches:
        start_index = match.end()
        end_index = start_index
        stack = 1

        # 从 \boxed{ 之后开始搜索，直到找到对应的闭合括号
        while stack > 0 and end_index < len(latex_str):
            if latex_str[end_index] == '{':
                stack += 1
            elif latex_str[end_index] == '}':
                stack -= 1
            end_index += 1

        if stack == 0:
            # 提取 \boxed{} 内部的内容
            content = latex_str[start_index:end_index - 1]
            results += content + ","
        else:
            # 如果括号没有正确闭合，则返回错误信息
            print("Mismatched braces in LaTeX string.")
            return ""

    return [results]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--answer_file_mm', type=str, default='answer.json')
    parser.add_argument('--answer_file_text', type=str, default='answer.json')
    # args
    args = parser.parse_args()

    # read results
    result_file_mm, result_file_text = read_jsonl(args.answer_file_mm), read_jsonl(args.answer_file_text)
    
    judger = MathJudger()
    
    def process_instance(d):
        if d['question_type'] == 'Theorem proof':
            return '', [False, 0]
        answer_type = d['answer_type']
        is_chinese = d['language'] == 'Chinese'
        model_answer = d['response']
        is_deepseek = False
        
        # model_answer = extract_answer(is_chinese, model_answer, is_deepseek)
        model_answer = extract_box(model_answer)
        if len(model_answer) == 0:
            return model_answer, [False, 1]
        else:
            model_answer = model_answer[0]
            ## too long, takes too much parsing time
            if len(model_answer) > 100:
                return model_answer, [False, 1]

        if 'Tuple' in answer_type: # 目前可机评的数据中 没有 need_human_evaluate
            judge_result = judger.judge(model_answer, d['final_answer'][0])
        else:
            if d['error']:
                if ',' in d['error']:
                    precisions = d['error'].split(',')
                    precisions = [float(p) if p else 1e-8 for p in precisions]
                    judge_result = judger.judge(model_answer, d['final_answer'][0], precisions)
                else:
                    precision = float(d['error'])
                    judge_result = judger.judge(model_answer, d['final_answer'][0], precision)
            else:
                judge_result = judger.judge(model_answer, d['final_answer'][0])
            
        return model_answer, [judge_result, 1]
    
    with Pool(processes=64) as pool:  # Specify number of processes
        processed_results_mm = list(tqdm(pool.imap(process_instance, result_file_mm), total=len(result_file_mm)))
    
    with Pool(processes=64) as pool:  # Specify number of processes
        processed_results_text = list(tqdm(pool.imap(process_instance, result_file_text), total=len(result_file_text)))
        
    acc1 = [pr[1] for pr in processed_results_mm]
    acc2 = [pr[1] for pr in processed_results_text]

    print("Text results:")
    print(np.array(acc1).sum(axis=0))
    print(np.array(acc1).sum(axis=0)[0] / np.array(acc1).sum(axis=0)[1])

    print("\n\nMM results:")
    print(np.array(acc2).sum(axis=0))
    print(np.array(acc2).sum(axis=0)[0] / np.array(acc2).sum(axis=0)[1])

    print("\n\nAll results:")
    acc = acc1.copy()
    acc.extend(acc2)
    print(np.array(acc).sum(axis=0))
    print(np.array(acc).sum(axis=0)[0] / np.array(acc).sum(axis=0)[1])
    
    
