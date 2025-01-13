import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils import *
from multiprocessing import Pool, Manager, cpu_count
import multiprocessing

# OpenAI
import openai

from prompts import demo_prompt_score


# load demo prompt
def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, inst):
    demo_prompt = demo_prompt.strip()
    full_prompt = demo_prompt.format(question = inst['question'], gt=inst['answer'], extraction=inst['extraction'])
    return full_prompt


def match_answer(inst, api_key, quick_match=False):
    # quick match
    if quick_match:
        return '1' if inst['answer'] == inst['extraction'] else '0'
    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt_score, inst)
        extraction = get_chat_response(full_prompt, api_key)
        return extraction.replace("Judgement:", "").strip()
    except Exception as e:
        print(e)
        print(f"Error in matching answer")

    return ""


def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = ' '.join(response.split(' ')[-trunk_length:])
        return return_res



def process_instance(args_tuple):
    """处理单个实例的函数，用于多进程"""
    i, inst, save_results, args = args_tuple
    save_inst = save_results[i] if i < len(save_results) else copy.deepcopy(inst)
    
    if args.cache and 'judgement' in save_inst:
        return save_inst
    else:
        judgement = match_answer(save_inst, args.api_key, args.quick_match)
        while True:
            if judgement.strip() not in ['0', '1']:
                print('Wrong return format: ', judgement)
                judgement = match_answer(save_inst, args.api_key, args.quick_match)
            else:
                save_inst['judgement'] = int(judgement)
                break

        return save_inst
    

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
    parser.add_argument('--answer_extraction_file', type=str, default='answer.json')
    parser.add_argument('--save_file', type=str, default='answer.json')
    # match
    parser.add_argument('--quick_match', action='store_true', help='use rules to match answer for some problems')
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    parser.add_argument('--cache', action='store_true', help='cache results')
    parser.add_argument('--trunk_response', type=int, default=-1, help='trunk response to the last n words')
    parser.add_argument('--api_key', type=str, help='api key for openai')
    # args
    args = parser.parse_args()

    # set api key
    openai.api_key = args.api_key

    # read results
    result_file = args.answer_extraction_file
    
    print(f"Reading {result_file}...")
    results = read_jsonl(result_file)
    score_version_dict = defaultdict(list)
    for res in results:
        ans = extract_box(res['response'])
        if len(ans) == 0:
            res['extraction'] = ""
            continue 
        res['extraction'] = ans[0]

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    if os.path.exists(args.save_file):
        save_results = json.load(open(args.save_file))
    else:
        save_results = []
        tasks = [(i, inst, save_results, args) for i, inst in enumerate(results)]
        # tqdm, enumerate results
        # 使用多进程处理
        with multiprocessing.Pool(processes=16) as pool:
            processed_results = list(tqdm(pool.imap(process_instance, tasks), total=len(tasks)))

    scores = [iss['judgement'] for iss in processed_results]
    total_cnt, right_cnt = 0, 0
    right_cnt = len([inst for inst in scores if inst == 1])
    total_cnt = len(scores)
    save_json(processed_results, args.save_file)
    
    print(f"Total Acc: {(right_cnt/total_cnt):.3f}")   