'''
author:        zhangjl19 <zhangjl19@spdb.com.cn>
date:          2024-03-11 11:10:40
'''
import sys
import os

def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


