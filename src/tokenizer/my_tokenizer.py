'''
author:        zhangjl19 <zhangjl19@spdb.com.cn>
date:          2024-03-11 11:14:45
'''
import sys
import os
import sentencepiece as spm
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_path, '../../'))
from src.tokenizer.tokenizer import Tokenizer


tokenizer_model_path = '/workspace/projects/llm/src/llama/configs/10w_vocab_wudao5_pile10.model'

sp_model = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
tokenizer = Tokenizer(sp_model)
