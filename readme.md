# intro
站在lamma 的肩膀上窥视大模型。
初期目标是从0-1构建一个能够执行的大模型全流程。

# 容器构建  
```
docker build -f torch2.0.1_cuda11.7.dockerfile -t llm-step-by-step:0.0.1
```

## 验证容器
### 构建容器
1. 交换式，后台常驻
docker run -dit --name=llm_step_by_step --shm-size 24g  --gpus all --runtime=nvidia -v /home/zhangjl19/:/workspace llm-step-by-step:0.0.1 /bin/bash

2. 进入容器
docker exec -it llm_step_by_step /bin/bash

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

# 参考资料
> https://github.com/meta-llama/llama/