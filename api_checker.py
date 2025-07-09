from openai import OpenAI
from collections import defaultdict
import pickle
import argparse
import json
import datetime
from time import sleep
import concurrent.futures
import json
import time
from tqdm import tqdm
from argparse import Namespace
import os

llm_to_api = {
    "gpt4": "gpt-4o",
    "mini" : "gpt-4o-mini",
    "gpt": "gpt-3.5-turbo-0125", 
    'claude': "claude-3-haiku-20240307",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "mixtral": "mixtral-8x7b-32768",
    "deepseek": "deepseek-chat",
    # "llama": "llama3-70b-8192",
    "llama8b": "meta-llama/Llama-3-8b-chat-hf",
    "llama": "meta-llama/Llama-3-70b-chat-hf",
    "qwen7b": "qwen1.5-7b-chat",
    "qwen": "qwen-plus-2024-09-19",
    "gemini": "gemini-1.5-pro",
    "gemma": "gemma-7b-it",
}


def process_data(i, data_answer, data_llm, llm, client, llm_to_api, response_dict, args):
    system_prompt = """You are a sophisticated AI. You are provided with the following:

1. A graph problem.  
2. The optimal solution to the problem: a1.  
3. The reasoning process for solving this problem concludes with the answer a2.  

Your task is to determine whether the optimal solution a1 aligns with the answer a2 derived from the reasoning process. If they align, output 'Yes'; otherwise, output 'No'. Only output 'Yes' or 'No' without providing any explanation."""
# 2. A hint to solve this problem.
    if args.resume and str(i) in response_dict and llm in response_dict[str(i)] and response_dict[str(i)][llm]:
        if response_dict[str(i)][llm]['output'] != 'Error!':
            print(i)
            return

    content = []
    content.append(f"###Problem###\n{data_answer['input']}\n")
    content.append(f"###The optimal answer:###\n{data_answer['hint']}\n")
    content.append(f"###The reasoning process:###\n{data_llm['output']}\n")
    llm_input = "\n".join(content)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": llm_input},
            ],
            model=llm_to_api[llm],
            seed=42,
            temperature=0.8
        )
        response_dict[str(i)][llm] = {}
        response_dict[str(i)][llm]['checker'] = chat_completion.choices[0].message.content
        print(llm, i, response_dict[str(i)][llm])
    except Exception as e:
        print('Call API failed! ', e)
        time.sleep(1)
        response_dict[str(i)][llm] = 'Error!'

    with open(f"results/tmp_{args.results}/{args.llm}.json", 'w') as f:
        json.dump(response_dict, f)

def main(args, datas_answer, datas_llm, llm, client, llm_to_api, response_dict):
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i in range(args.st, args.ed):
            data_answer = datas_answer[i]
            data_llm = datas_llm[str(i)]['deepseek']
            response_dict[str(i)] = {}
            futures.append(executor.submit(process_data, i, data_answer, data_llm, llm, client, llm_to_api, response_dict, args))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    # 等待所有任务完成后，休眠一段时间
    time.sleep(args.sleep)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt', help='llm model name')
    parser.add_argument('--task', type=str, default='TSP', help='task name')
    parser.add_argument('--problem_num', type=int, default=10, help='number of problems')
    parser.add_argument('--example_num', type=int, default=2, help='number of examples')
    parser.add_argument('--difficulty', type=str, default='easy', help='problem difficulty')
    parser.add_argument('--resume', type=bool, default=False, help='resume from last checkpoint')
    parser.add_argument('--results', type=str, default='tmp', help='results location')
    parser.add_argument('--sleep', type=int, default=5, help='sleep seconds between API calls')
    parser.add_argument('--st', type=int, default=0, help='start index')
    parser.add_argument('--ed', type=int, default=0, help='end index')
    parser.add_argument('--num_workers', type=int, default=64, help='Number of workers in the thread pool')

    args = parser.parse_args()
    error_knt = 0
    
    response_dict = defaultdict(dict)
    
    cnt = 0
    for llm in args.llm.split('-'):
        if 'gpt' in llm or 'mini' in llm:
            client = OpenAI(
                base_url = "https://35.aigcbest.top/v1",
                #api_key = 'sk-JMe1osu1CZFTcEx942Cf36A188Cc49D6Ab684689F05e3fE7',
                api_key = 'sk-4PhY006ulJWLVjUf19B39a8f9bB34aBdA7B0C21e96B38873' 
            )
        elif llm == 'deepseek':
            client = OpenAI(
                base_url = "https://api.deepseek.com",
                api_key = 'sk-6eada9420c23459b8aa28ceabfa4f9e6'
            )
        elif 'llama' in llm or 'mixtral' in llm:
            client = OpenAI(
                base_url = "https://api.aimlapi.com/",
                api_key = '23577ec496f14b7ebf00767d7ea0cd3a'
            )
        elif 'qwen' in llm:
            client = OpenAI(
                api_key="sk-95cef5180dec49ac8dc40936fa3b3548",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            client = OpenAI(
                base_url = "https://aigcbest.top/v1",
                api_key = 'sk-995MSFbYANjPjjvI9d1d331eD3Ac4d55B4BeBe1b400052Ae'
            )

        if args.resume and os.path.exists(f"results/tmp_{args.results}/{args.llm}.json"):
            with open(f"results/tmp_{args.results}/{args.llm}.json", 'r') as f:
                response_dict = json.load(f)
                print(f"Continue")

        if not os.path.exists(f"results/tmp_{args.results}"):
            os.makedirs(f"results/tmp_{args.results}")
            
        with open(f"/mnt/sdc/qifan/Graphwiz_Modify/Prompt_api_data_0104/{args.task}.json","r") as f:
            datas_answer = json.load(f)
        
        with open(f"/mnt/sdc/qifan/Graphwiz_Modify/Final_datas_0104/{args.task}.json","r") as f:
            datas_llm = json.load(f)
        
        # print(len(datas))
        # if len(datas) > 30000:
        #     datas = datas[:30000]
        args.ed = min(len(datas_answer), len(datas_llm))
        print(len(datas_answer), len(datas_llm))
        main(args, datas_answer,datas_llm, llm, client, llm_to_api, response_dict)
        
    print('error_knt:', error_knt)
    now = datetime.datetime.now()
    if not os.path.exists(f"results/{args.results}"):
        os.makedirs(f"results/{args.results}")
    with open(f"results/{args.results}/{args.llm}_{args.task}_{args.difficulty}_{now.strftime('%d_%H-%M')}.json", 'w') as f:
        json.dump(response_dict, f)
        