# intro
站在lamma 的肩膀上窥视大模型。

# 容器构建
docker build -f torch2.0.1_cuda11.7.dockerfile -t llm-step-by-step:0.0.1



# 代理
## 代理设置
export http_proxy=http://192.168.200.26:51837  
export https_proxy=http://192.168.200.26:51837  

## 取消代理
unset https_proxy  
unset http_proxy  