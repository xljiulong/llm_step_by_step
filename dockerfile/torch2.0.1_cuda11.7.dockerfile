FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

LABEL maintainer="zhangjl19@spdb.com.cn"
LABEL version = "torch2.0.1_cuda11.7"
LABEL description = "torch2.0.1_cuda11.7-runtime for llm-step-by-step environments"

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    echo 'Asia/Shanghai' > /etc/timezone && \
    apt update && \
    apt install -y git && \
    apt install -y openssh-server && \
    apt install -y pdsh && \
    apt install -y net-tools && \
    apt install -y iputils-ping && \
    apt install -y vim
