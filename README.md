# VoiceInteractionAgent

超级精简的语音交互项目：后端（FastAPI WebSocket）+ 可选 GPT-SoVITS(开源项目) 语音克隆；默认也可用 qwen3-tts。详情请查看内部的readme

## 结构
- `backend_realtime/`：实时语音对话后端（ASR → LLM → TTS）。
- `GPT-SoVITS/`：可选语音克隆服务；未启用时后端可直接用 qwen3-tts。

## 快速开始
1. 后端
   ```bash
   cd backend_realtime
   cp .env.example .env   # 按需填 API Key/模型/音色
   pip install -r requirements.txt
   bash start_server.sh   # 默认 0.0.0.0:8044
   ```
2. 可选 GPT-SoVITS
   ```bash
   cd GPT-SoVITS
   pip install -r requirements.txt
   bash start.sh          # 启动克隆服务，后端即可用 gptsovits 引擎
   ```

## 前端需要实现的功能
- 1. 智能语音接口下支持语音/文本/图片的多模态输入。
- 2. 实时翻译接口
- 3. 实现语音唤醒与语音智能截断功能(这部分要在前端做，目前已经有成熟的方案)


## 设计提醒
与普通对话窗口不同，智能语音犹如小爱同学，并没有窗口转换新建聊天窗口的实现需求(当然这个也能实现，具体请查看GLOBAL_SESSION_ID参数的功能)文档记录我是使用本地单文件sqlite进行存储