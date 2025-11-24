#!/bin/bash
# 启动 WebSocket 后端服务，支持大消息传输

source /home/data/nongwa/miniconda3/bin/activate intvosys

# 设置 WebSocket 最大消息大小为 10MB
export WEBSOCKET_MAX_SIZE=10485760



# 启动服务
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8044 \
    --reload \
    --ws-max-size 10485760
