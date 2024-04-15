'''
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-16 22:10:44
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-26 22:59:55
FilePath: /Open-Llama/data/preprocess_wudao.py
Description: 
Parse the dataset from the raw files and split them into different jsonl files based on the preset maximum number of lines, 
making it easy for parallel training to perform streaming reads.
Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
'''
import os
import json
from glob import glob
from tqdm import tqdm
import zstandard as zstd
from pathlib import Path
import argparse

def process_wudao(wudao_path, wudao_dst_path, record_num_per_file):
    # wudao_path = Path(f'{wudao_path}/part*')
    wudao_p = Path(wudao_path)
    wudao_files = wudao_p.glob('part*') # glob('/workspace/projects/Open-Llama/data/WuDaoCorpus2.0_base_200G/part*')
    
    # 创建目标目录
    wudao_dstp = Path(wudao_dst_path)
    if not wudao_dstp.exists():
        print(f'creating wudao dst path {wudao_dst_path}')
        wudao_dstp.mkdir(parents=True)
        
    write_path = os.path.join(wudao_dst_path, 'part-wudao-{}.jsonl.zst')
        
    total_num = 0
    file_num = 0
    wfp = zstd.open(write_path.format(file_num), 'wb', encoding='utf-8')
    wudao_files_lst = list(wudao_files)
    for tpath in tqdm(wudao_files_lst, total=len(wudao_files_lst)):
        # print(f'processing file {tpath}')
        with open(tpath, 'r') as fp:
            data = json.load(fp)
        for line in data:
            if total_num % record_num_per_file == 0 and total_num > 0:
                file_num += 1
                wfp.close()
                wfp = zstd.open(write_path.format(file_num), 'wb', encoding='utf-8')
            wfp.write(json.dumps(line).encode('utf-8'))
            wfp.write('\n'.encode('utf-8'))
            total_num += 1
    wfp.close()
    print('total line: {}\ntotal files: {}'.format(total_num, file_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ttsing")
    parser.add_argument("--wudao_path", type=str, default='/workspace/projects/Open-Llama/data/WuDaoCorpus2.0_base_200G/', help="指定wudao数据目录路径")
    parser.add_argument("--wudao_write_path", type=str, default='/workspace/projects/llm_step_by_step/data/pretrain_data/', help="wudao数据处理后的目录")
    parser.add_argument("--record_num_per_file", type=int, default=16384, help="每个zst文件记录数量")
    
    args = parser.parse_args()
    wudao_path = args.wudao_path
    wudao_write_path = args.wudao_write_path
    record_num_per_file = args.record_num_per_file
    process_wudao(wudao_path, wudao_write_path, record_num_per_file)
