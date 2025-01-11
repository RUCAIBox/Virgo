import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils import *
import time
# OpenAI
import openai
import multiprocessing

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

def extract_answer(response, trunk_num):
    response = response.strip().split("\n")[-1]
    prefixes = [
        "answer is", "Answer", "Answer:", "Correct option", "the correct option is", 
        "The correct option is", "the correct statement is", 
        "Final answer", "the answer is", "is:\n\n", "the answer is", "correct answer is", "Option:", "correct statement is", "\n\n"
    ]
    for prefix in prefixes:
        if prefix in response:
            response = response.split(prefix)[-1].strip()
    response = response.lstrip(":").strip().split(" ")[0]
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    return trunk_response(response, trunk_num)

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
    print(f"================================{args.answer_extraction_file}==============================")

    # set api key
    openai.api_key = args.api_key

    # read results
    result_file = args.answer_extraction_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)
    # results = [json.loads(line) for line in open(result_file, 'r').readlines()]
    # for ele in results:
    #     if "\\boxed{" in ele['response']:
    #         ele['response'] = ele['response'].split("\\boxed{")[-1].split("}")[0]
    #     if args.quick_match:
    #         ele['extraction'] = extract_answer(ele['response'], args.trunk_response)
    #     else:
    #         ele['extraction'] = ele['response']

    score_version_dict = defaultdict(list)
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    if os.path.exists(args.save_file):
        save_results = json.load(open(args.save_file))
        for save_inst in save_results:
            score_version_dict[save_inst['problem_version']].append(save_inst['judgement'])
    else:
        save_results = []
        # 创建任务列表
        tasks = [(i, inst, save_results, args) for i, inst in enumerate(results)]
        # tqdm, enumerate results
        # 使用多进程处理
        with multiprocessing.Pool(processes=32) as pool:
            processed_results = list(tqdm(pool.imap(process_instance, tasks), total=len(tasks)))
        # 更新 save_results 和 score_version_dict
        for i, save_inst in enumerate(processed_results):
            save_results.append(save_inst)
            score_version_dict[save_inst['problem_version']].append(save_inst['judgement'])
        # 保存结果
        save_json(save_results, args.save_file)

    # version level acc
    total_cnt, right_cnt = 0, 0
    for version in score_version_dict:
        version_total_cnt = len(score_version_dict[version])
        version_right_cnt = len([inst for inst in score_version_dict[version] if inst == 1])
        total_cnt += version_total_cnt
        right_cnt += version_right_cnt
        print(f"{version} Acc: {(version_right_cnt/version_total_cnt):.3f}")
        print(version_total_cnt)

    print(f"Acc: {(right_cnt/total_cnt):.3f}")     