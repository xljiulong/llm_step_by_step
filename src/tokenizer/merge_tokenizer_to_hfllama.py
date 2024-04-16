import os
import argparse

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from pathlib import Path
 
def merge_tokenizer(args):
    llama2_tokenizer_dir = args.llama_tokenizer_model
    llama2_tokenizer = LlamaTokenizer.from_pretrained(llama2_tokenizer_dir)
    
    sf_sp_model = spm.SentencePieceProcessor()
    sf_sp_model_file = args.sf_tokenizer_model
    sf_sp_model.Load(sf_sp_model_file)
    
    llama2_spm = sp_pb2_model.ModelProto()
    llama2_spm.ParseFromString(llama2_tokenizer.sp_model.serialized_model_proto())
    
    sf_spm = sp_pb2_model.ModelProto()
    sf_spm.ParseFromString(sf_sp_model.serialized_model_proto())
    
    
    # print number of tokens
    print(len(llama2_tokenizer), len(sf_sp_model))
    print(llama2_tokenizer.all_special_tokens)
    print(llama2_tokenizer.all_special_ids)
    print(llama2_tokenizer.special_tokens_map)
    
    ## Add Chinese tokens to LLaMA2 tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama2_spm.pieces)
    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    
    for p in sf_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama2_spm.pieces.append(new_p)
    print(f"New model pieces: {len(llama2_spm.pieces)}")
    
    ## Save
    output_sp_dir = args.merged_tokenizer_dir
    output_path = Path(output_sp_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        
    # os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + '/merged_llama2.model', 'wb') as f:
        f.write(llama2_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/merged_llama2.model')
    try:
        Path(output_sp_dir + '/merged_llama2.model').unlink()
    except Exception as e:
        print("Error: deleting %s : %s" % ('merged_llama2.model', e.strerror))
    
    output_hf_dir = output_sp_dir #'llama2_chinese'  #
    os.makedirs(output_hf_dir, exist_ok=True)
    tokenizer.save_pretrained(output_hf_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")
    
    # Test
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama2_tokenizer_dir)
    merged_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(f"Tokenized by ChatGLM tokenizer:{merged_llama_tokenizer.tokenize(text)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ttsing")
    parser.add_argument("--llama_tokenizer_model", type=str, default='/workspace/models/huggingface/LLaMA2-7B-ZH-Chat-52k/tokenizer.model', help="指定被合并的目标tokenizer(llama)")
    parser.add_argument("--sf_tokenizer_model", type=str, default='/workspace/projects/llm/src/llama/configs/2w_vocab_wudao5_pile10.model', help="指定待合并的tokenizer(新tokenizer)")
    parser.add_argument("--merged_tokenizer_dir", type=str, default='/workspace/models/huggingface/merged_tokenizer', help="合并后的tokenizer 目录")
    
    args = parser.parse_args()
    merge_tokenizer(args)