
'''
Author: zhangjiulong
email: xljiulong@gmail.com
'''

import json
from glob import glob
import zstandard as zstd


def create_data_iter(paths, transform_dict=None, process_index=0, num_processes=1):
    '''
    Currently, the allowed storage formats are jsonl and jsonl.zst. 
    Each line of the data is a dictionary, which can be parsed as JSON for subsequent processing after reading.
    '''
    past = None
    for i, path in paths:
        dataset_name = path.split('-')[-2]
        if past != dataset_name:
            print('Loading data from {}'.format(path))
            past = path
        if num_processes > 1 and i % num_processes != process_index:
            continue
        if path.endswith('jsonl.zst'):
             with zstd.open(path, 'r', encoding='utf-8') as fp:
                 for line in fp:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    line = json.loads(line)
                    line['dataset'] = dataset_name
                    if transform_dict:
                        line = transform_dict[dataset_name](line)
                        if isinstance(line, str):
                            yield line
                        elif isinstance(line, list):
                            for i in line:
                                yield i
                        else:
                            raise Exception('Unsupported type in Transformation: {}'.format(transform_dict[dataset_name]))
                    else:
                        yield line
        elif path.endswith('jsonl'):
            with open(path, 'r') as fp:
                for line in fp:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    line = json.loads(line)
                    line['dataset'] = dataset_name
                    if transform_dict:
                        line = transform_dict[dataset_name](line)
                        if isinstance(line, str):
                            yield line
                        elif isinstance(line, list):
                            for i in line:
                                yield i
                        else:
                            raise Exception('Unsupported type in Transformation: {}'.format(transform_dict[dataset_name]))
                    else:
                        yield line
        else:
            raise Exception('File format of {} is not supported yet.'.format(path))

def create_shard_kwargs(patterns, repeat=1):
    '''
    Assign numbers to different shards of data to ensure that data is not duplicated 
    when allocated to different nodes during distributed training.
    '''
    all_path = []
    for p in patterns:
        all_path.extend(glob(p))
    all_path *= repeat
    return [(i, p) for i, p in enumerate(all_path)]

if __name__ == '__main__':
    patterns = [
        'data/pretrain_data/part-wudao*.jsonl.zst'
    ]
    paths = create_shard_kwargs(patterns)
    transform_dict = {
        'wudao': lambda x: x['title'],
        'pile': lambda x: [x['text']]
    }
    data_iter = create_data_iter(paths, transform_dict=transform_dict)
    for i, data in enumerate(data_iter):
        print(i, data)
        if i == 20:
            break