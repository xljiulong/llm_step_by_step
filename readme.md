# Intro
站在llama 的肩膀上窥视大模型。
初期目标是从0-1构建一个能够执行的大模型全流程。

# 容器构建  
```
docker build -f torch2.0.1_cuda11.7.dockerfile -t llm-step-by-step:0.0.1
```

## 验证容器
1. 交互式，后台常驻  
```
docker run -dit --name=llm_step_by_step --shm-size 24g  --gpus all --runtime=nvidia -v /home/zhangjl19/:/workspace llm-step-by-step:0.0.1 /bin/bash
```

2. 进入容器  
```
docker exec -it llm_step_by_step /bin/bash
```

# docker compose
1. 一键式构建
```
docker-compose up -d --build
docker compose up -d --build
```

# tokenizer  
## 数据处理  
### 构建一个方便分布式训练的数据集
```
python preprocess_wudao.py 
```

## 训练tokenizer
```
python train_tokenizer.py
```

## 扩充tokenizer  
```
python merge_tokenizer_to_hfllama.py
```

# 参考资料
> https://github.com/meta-llama/llama/  
> https://github.com/yangjianxin1/  
> https://github.com/ymcui/Chinese-LLaMA-Alpaca-2  
> https://github.com/yangjianxin1/LLMPruner/tree/master 根据token重新生成模型  
> https://zhuanlan.zhihu.com/p/672712751 moe
> https://github.com/pjlab-sys4nlp/llama-moe/tree/main LLaMa MOE
> https://zhuanlan.zhihu.com/p/649756898 
> https://www.bilibili.com/video/BV12h4y1N7C8/?spm_id_from=333.788.recommend_more_video.0&vd_source=64f63f34985a708ab738d22e9d0dd177 知乎 哔哩哔哩 博主