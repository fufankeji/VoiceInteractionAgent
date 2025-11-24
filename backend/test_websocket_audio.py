"""测试 WebSocket 音频接口"""
import asyncio
import base64
import json
import wave
from pathlib import Path

import websockets


async def test_websocket_audio():
    """测试通过 WebSocket 发送音频文件"""

    # WebSocket 服务器地址
    ws_url = "ws://192.168.110.131:8044/ws?session_id=test_user"

    # 音频文件路径（请根据实际情况修改）
    audio_file = Path("/home/data/nongwa/workspace/model/TTS-GPT_SoVITS-sunshine_girl/sunshine_girl.wav")

    # 如果没有测试音频文件，生成一个简单的 WAV 文件
    if not audio_file.exists():
        print(f"⚠️  测试音频文件不存在: {audio_file}")
        print("💡 请提供一个 WAV 格式的音频文件，或修改 audio_file 变量指向实际文件路径")
        return

    # 读取音频文件并编码为 base64
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    print(f"📁 读取音频文件: {audio_file}")
    print(f"📊 音频大小: {len(audio_bytes)} 字节")
    print(f"🔗 连接到: {ws_url}")

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=20,  # 每20秒发送一次 ping
            ping_timeout=10,   # ping 超时时间
            close_timeout=10,  # 关闭超时时间
            max_size=10 * 1024 * 1024  # 最大消息大小 10MB
        ) as websocket:
            print("✅ WebSocket 连接成功")

            # 构造音频消息
            message = {
                "type": "audio",
                "data": audio_b64
            }

            # 发送音频数据
            print("📤 发送音频数据...")
            await websocket.send(json.dumps(message))
            print("✅ 音频已发送")

            # 接收服务器响应
            print("\n📥 等待服务器响应...\n")

            audio_chunks = []
            full_text = ""
            asr_result = ""

            while True:
                try:
                    # 增加超时时间，给 ASR 和 LLM 更多处理时间
                    response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    data = json.loads(response)
                    msg_type = data.get("type")

                    if msg_type == "asr_result":
                        asr_result = data.get("text", "")
                        print(f"🎤 [ASR识别] {asr_result}")

                    elif msg_type == "audio":
                        audio_data = data.get("data", "")
                        audio_chunks.append(audio_data)
                        print(f"🔊 [音频块] 收到第 {len(audio_chunks)} 个音频块")

                    elif msg_type == "final_text":
                        full_text = data.get("text", "")
                        print(f"\n💬 [完整回复] {full_text}")
                        # 不要立即退出，继续接收可能的音频块

                    elif msg_type == "audio_end":
                        print("\n✅ 音频传输完成，准备退出...")
                        break

                    elif msg_type == "error":
                        error_msg = data.get("error", "")
                        print(f"❌ [错误] {error_msg}")
                        break

                    else:
                        print(f"📨 [其他消息] type={msg_type}, data={data}")

                except asyncio.TimeoutError:
                    print("⏰ 等待超时，可能服务器处理较慢或已完成")
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"⚠️  连接已关闭: {e}")
                    break
                except Exception as e:
                    print(f"❌ 接收消息时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            # 保存接收到的音频
            if audio_chunks:
                output_file = Path("received_audio.wav")
                combined_audio = base64.b64decode("".join(audio_chunks))

                with open(output_file, "wb") as f:
                    f.write(combined_audio)

                print(f"\n💾 已保存接收到的音频: {output_file} ({len(combined_audio)} 字节)")

            # 打印测试总结
            print("\n" + "="*60)
            print("📊 测试总结")
            print("="*60)
            print(f"输入音频: {audio_file}")
            print(f"ASR 识别结果: {asr_result or '无'}")
            print(f"AI 回复文本: {full_text or '无'}")
            print(f"接收音频块数: {len(audio_chunks)}")
            print("="*60)

    except ConnectionRefusedError:
        print("❌ 连接失败: 服务器未启动或地址错误")
        print("💡 请确保后端服务已启动: uvicorn app.main:app --reload --port 8000")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_websocket_text():
    """测试通过 WebSocket 发送文本消息"""

    ws_url = "ws://192.168.110.131:8044/ws?session_id=test_user"

    print(f"🔗 连接到: {ws_url}")

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
            max_size=10 * 1024 * 1024  # 最大消息大小 10MB
        ) as websocket:
            print("✅ WebSocket 连接成功")

            # 构造文本消息
            message = {
                "type": "text",
                "text": "今天星期几？"
            }

            # 发送文本数据
            print(f"📤 发送文本: {message['text']}")
            await websocket.send(json.dumps(message))
            print("✅ 消息已发送")

            # 接收服务器响应
            print("\n📥 等待服务器响应...\n")

            audio_chunks = []
            full_text = ""

            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    data = json.loads(response)
                    msg_type = data.get("type")

                    if msg_type == "audio":
                        audio_data = data.get("data", "")
                        audio_chunks.append(audio_data)
                        print(f"🔊 [音频块] 收到第 {len(audio_chunks)} 个音频块")

                    elif msg_type == "final_text":
                        full_text = data.get("text", "")
                        print(f"\n💬 [完整回复] {full_text}")
                        # 不要立即退出，继续接收音频

                    elif msg_type == "audio_end":
                        print("\n✅ 音频传输完成")
                        break

                    elif msg_type == "error":
                        error_msg = data.get("error", "")
                        print(f"❌ [错误] {error_msg}")
                        break

                    else:
                        print(f"📨 [其他消息] type={msg_type}")

                except asyncio.TimeoutError:
                    print("⏰ 等待超时")
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"⚠️  连接已关闭: {e}")
                    break
                except Exception as e:
                    print(f"❌ 接收消息时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            # 保存接收到的音频
            if audio_chunks:
                output_file = Path("received_text_response.wav")
                combined_audio = base64.b64decode("".join(audio_chunks))

                with open(output_file, "wb") as f:
                    f.write(combined_audio)

                print(f"\n💾 已保存接收到的音频: {output_file} ({len(combined_audio)} 字节)")

            # 打印测试总结
            print("\n" + "="*60)
            print("📊 测试总结")
            print("="*60)
            print(f"输入文本: {message['text']}")
            print(f"AI 回复: {full_text or '无'}")
            print(f"接收音频块数: {len(audio_chunks)}")
            print("="*60)

    except ConnectionRefusedError:
        print("❌ 连接失败: 服务器未启动或地址错误")
        print("💡 请确保后端服务已启动: uvicorn app.main:app --reload --port 8000")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def create_test_audio():
    """创建一个简单的测试音频文件（1秒的静音）"""
    output_file = Path("test_audio.wav")

    if output_file.exists():
        print(f"⚠️  测试音频已存在: {output_file}")
        return

    # 参数
    sample_rate = 16000  # 16kHz
    duration = 1  # 1秒
    channels = 1  # 单声道
    sample_width = 2  # 16-bit

    # 生成静音数据
    num_samples = sample_rate * duration
    silence = b'\x00\x00' * num_samples

    # 写入 WAV 文件
    with wave.open(str(output_file), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(silence)

    print(f"✅ 已创建测试音频文件: {output_file}")
    print(f"   - 采样率: {sample_rate} Hz")
    print(f"   - 时长: {duration} 秒")
    print(f"   - 声道: {channels}")
    print(f"   - 位深: {sample_width * 8} bit")


if __name__ == "__main__":
    import sys

    print("="*60)
    print("🎙️  WebSocket 音频接口测试工具")
    print("="*60)
    print()
    print("测试模式:")
    print("  1. 测试文本输入 (推荐先测试)")
    print("  2. 测试音频输入")
    print("  3. 创建测试音频文件")
    print()

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = input("请选择测试模式 (1/2/3): ").strip()

    if mode == "1":
        print("\n🚀 开始测试文本输入...\n")
        asyncio.run(test_websocket_text())

    elif mode == "2":
        print("\n🚀 开始测试音频输入...\n")
        asyncio.run(test_websocket_audio())

    elif mode == "3":
        print("\n🚀 创建测试音频文件...\n")
        create_test_audio()

    else:
        print("❌ 无效的选择，请输入 1、2 或 3")
