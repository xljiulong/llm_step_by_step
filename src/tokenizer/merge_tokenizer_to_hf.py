import os
 
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
 
llama2_tokenizer_dir = "llama2_tokenizer/tokenizer.model"
llama2_tokenizer = LlamaTokenizer.from_pretrained(llama2_tokenizer_dir)
 
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model_file = "tokenizer.model"
chinese_sp_model.Load(chinese_sp_model_file)
 
llama2_spm = sp_pb2_model.ModelProto()
llama2_spm.ParseFromString(llama2_tokenizer.sp_model.serialized_model_proto())
 
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
 
 
# print number of tokens
print(len(llama2_tokenizer), len(chinese_sp_model))
print(llama2_tokenizer.all_special_tokens)
print(llama2_tokenizer.all_special_ids)
print(llama2_tokenizer.special_tokens_map)
 
## Add Chinese tokens to LLaMA2 tokenizer
llama_spm_tokens_set = set(p.piece for p in llama2_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
 
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama2_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama2_spm.pieces)}")
 
## Save
output_sp_dir = 'llama2_chinese'
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + '/chinese_llama2.model', 'wb') as f:
    f.write(llama2_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/chinese_llama2.model')
 
output_hf_dir = 'llama2_chinese'  #
os.makedirs(output_hf_dir, exist_ok=True)
tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")
 
# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama2_tokenizer_dir)
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
The primary use of LLaMA is research on large language models, including'''
print("Test text:\n", text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by ChatGLM tokenizer:{chinese_llama_tokenizer.tokenize(text)}")