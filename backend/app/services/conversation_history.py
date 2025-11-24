"""对话历史管理模块 - 借鉴 MoeChat 的双层存储架构"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from ruamel.yaml import YAML

from app.utils.text_cleaner import strip_stage_directions


class ConversationHistory:
    """
    对话历史管理器
    
    双层存储架构:
    1. history.yaml - 短期上下文 (最近 N 条,用于 LLM)
    2. memory/YYYY-MM-DD.yaml - 长期记忆 (按日期归档,用于检索)
    """
    
    def __init__(
        self,
        agent_id: str = "default",
        data_dir: str = "./data/conversations",
        max_context_length: int = 20,
    ):
        self.agent_id = agent_id
        self.data_dir = Path(data_dir) / agent_id
        self.max_context_length = max_context_length
        
        # 创建目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.data_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)
        
        # 文件路径
        self.history_file = self.data_dir / "history.yaml"
        
        # 内存缓存
        self.context: List[Dict] = []
        self.temp_buffer: List[Dict] = []
        
        # 加载历史
        self._load_history()

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[History][{timestamp}] {message}")
    
    def _load_history(self):
        """启动时加载历史上下文"""
        if not self.history_file.exists():
            return
        
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                all_messages = yaml.safe_load(f) or []
            
            # 只加载最近的 N 条
            self.context = all_messages[-self.max_context_length:]
            self._log(f"加载了 {len(self.context)} 条历史对话")
        except Exception as e:
            self._log(f"加载历史失败: {e}")
    
    def add_message(self, role: str, content: str, save_immediately: bool = False):
        """
        添加一条消息
        
        Args:
            role: "user" 或 "assistant"
            content: 消息内容
            save_immediately: 是否立即保存 (默认 False,等到 save() 时批量保存)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 格式化消息 (嵌入时间信息,学习 MoeChat)
        formatted_content = f"\n<当前时间>{timestamp}</当前时间>\n"
        if role == "user":
            formatted_content += f"<用户对话内容>\n{content}\n</用户对话内容>\n"
        else:
            formatted_content = content  # assistant 直接用原内容
        
        final_content = formatted_content if role == "user" else strip_stage_directions(content)
        message = {
            "role": role,
            "content": final_content,
        }
        
        # 添加到上下文和临时缓冲
        self.context.append(message)
        self.temp_buffer.append(message)
        
        # 保持上下文长度
        if len(self.context) > self.max_context_length:
            self.context = self.context[-self.max_context_length:]
        
        if save_immediately:
            self.save()
    
    def save(self):
        """
        保存对话到文件
        
        双层保存:
        1. 追加到 history.yaml (短期上下文)
        2. 归档到 memory/日期.yaml (长期记忆)
        """
        if not self.temp_buffer:
            return
        
        try:
            # 1. 保存到 history.yaml (追加模式)
            yaml_writer = YAML()
            yaml_writer.indent(mapping=2, sequence=4, offset=2)
            yaml_writer.default_flow_style = False
            yaml_writer.allow_unicode = True
            
            with open(self.history_file, "a", encoding="utf-8") as f:
                yaml_writer.dump(self.temp_buffer, f)
            
            # 2. 归档到长期记忆 (按日期分文件)
            self._save_to_memory()
            
            self._log(f"保存了 {len(self.temp_buffer)} 条对话")
            self.temp_buffer = []
            
        except Exception as e:
            self._log(f"保存失败: {e}")
    
    def _save_to_memory(self):
        """保存到长期记忆 (按日期归档)"""
        today = datetime.now().strftime("%Y-%m-%d")
        memory_file = self.memory_dir / f"{today}.yaml"
        
        # 构建记忆条目 (学习 MoeChat 的格式)
        memory_entries = {}
        
        # 按对话回合组织 (user + assistant)
        i = 0
        while i < len(self.temp_buffer):
            msg = self.temp_buffer[i]
            
            # 生成时间戳 key
            timestamp = int(time.time())
            
            if msg["role"] == "user" and i + 1 < len(self.temp_buffer):
                # 完整回合: user + assistant
                user_msg = msg["content"]
                assistant_msg = self.temp_buffer[i + 1]["content"]
                
                memory_entries[timestamp] = {
                    "text_tag": "对话",  # 可以后续添加分类逻辑
                    "msg": f"用户: {user_msg}\n助手: {assistant_msg}",
                }
                i += 2
            else:
                # 单条消息
                memory_entries[timestamp] = {
                    "text_tag": msg["role"],
                    "msg": msg["content"],
                }
                i += 1
            
            time.sleep(0.001)  # 确保时间戳不重复
        
        # 追加到日期文件
        existing_data = {}
        if memory_file.exists():
            with open(memory_file, "r", encoding="utf-8") as f:
                existing_data = yaml.safe_load(f) or {}
        
        existing_data.update(memory_entries)
        
        yaml_writer = YAML()
        yaml_writer.indent(mapping=2, sequence=2, offset=2)
        yaml_writer.default_flow_style = False
        yaml_writer.allow_unicode = True
        
        with open(memory_file, "w", encoding="utf-8") as f:
            yaml_writer.dump(existing_data, f)
    
    def get_context(self) -> List[Dict]:
        """获取当前上下文 (用于 LLM)"""
        return self.context
    
    def search_by_date(self, date: str) -> Optional[Dict]:
        """
        根据日期搜索记忆
        
        Args:
            date: YYYY-MM-DD 格式
        
        Returns:
            该日期的所有对话记录
        """
        memory_file = self.memory_dir / f"{date}.yaml"
        
        if not memory_file.exists():
            return None
        
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self._log(f"读取记忆失败: {e}")
            return None
    
    def search_recent(self, days: int = 7) -> List[Dict]:
        """
        搜索最近 N 天的对话
        
        Returns:
            按时间戳排序的对话列表
        """
        from datetime import timedelta
        
        results = []
        today = datetime.now()
        
        for i in range(days):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            memories = self.search_by_date(date)
            
            if memories:
                for timestamp, data in memories.items():
                    results.append({
                        "timestamp": timestamp,
                        "date": date,
                        "tag": data.get("text_tag", ""),
                        "content": data.get("msg", ""),
                    })
        
        # 按时间戳排序
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results
    
    def clear_context(self):
        """清空当前上下文 (不删除历史文件)"""
        self.context = []
        self.temp_buffer = []
        print("[History] 上下文已清空")


# 使用示例
if __name__ == "__main__":
    # 创建历史管理器
    history = ConversationHistory(
        agent_id="user_001",
        max_context_length=20,
    )
    
    # 添加对话
    history.add_message("user", "你好,今天天气怎么样?")
    history.add_message("assistant", "今天阳光明媚,非常适合出门散步!")
    
    history.add_message("user", "那我们一起去公园吧")
    history.add_message("assistant", "好呀!我会陪着你的~")
    
    # 保存
    history.save()
    
    # 获取上下文
    context = history.get_context()
    print(f"\n当前上下文: {len(context)} 条")
    for msg in context:
        print(f"  [{msg['role']}] {msg['content'][:50]}...")
    
    # 搜索历史
    today = datetime.now().strftime("%Y-%m-%d")
    memories = history.search_by_date(today)
    print(f"\n今天的记忆: {len(memories) if memories else 0} 条")
    
    # 搜索最近 7 天
    recent = history.search_recent(days=7)
    print(f"\n最近 7 天: {len(recent)} 条对话")
