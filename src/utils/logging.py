'''
author:        zhangjl19 <zhangjl19@spdb.com.cn>
date:          2024-03-11 11:05:13
'''
import sys
import os
import logging
import loguru
from typing import List


logger = loguru.logger

def log_dist(message: str,
             ranks: List[int] = [],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')