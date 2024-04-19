FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="zhangjl19@spdb.com.cn"
LABEL version = "torch2.0.1_cuda11.7"
LABEL description = "torch2.0.1_cuda11.7-runtime for llm-step-by-step environments"

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    echo 'Asia/Shanghai' > /etc/timezone && \
    apt-get update && \
    apt-get install -y git && \
    apt-get install -y openssh-server && \
    apt-get install -y pdsh && \
    apt-get install -y net-tools && \
    apt-get install -y iputils-ping && \
    apt-get install -y vim && \
    pip install --no-cache-dir -r /tmp/requirements.txt
