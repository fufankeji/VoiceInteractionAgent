"""
长期记忆系统 - 为语音助手添加对话历史记忆能力

借鉴 MoeChat 的设计:
1. 时间戳索引的对话历史
2. 基于时间的快速检索
3. 向量相似度增强检索
4. 核心记忆独立存储
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta


class SimpleMemorySystem:
    """
    简化的长期记忆系统
    
    特性:
    1. 对话历史自动保存 (按日期分文件)
    2. 时间范围快速检索 (二分查找)
    3. 核心记忆单独管理
    """
    
    def __init__(self, memory_dir: str = "data/memories"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 历史记忆索引: {timestamp: message}
        self.memory_index: List[float] = []  # 排序的时间戳列表
        self.memory_data: Dict[float, Dict] = {}  # 时间戳 -> 对话数据
        
        # 核心记忆
        self.core_memories: List[str] = []
        
        # 加载历史
        self._load_memories()
        self._load_core_memories()
    
    def _load_memories(self):
        """加载最近 30 天的对话历史"""
        print("[记忆系统] 加载对话历史...")
        
        count = 0
        # 加载最近 30 天的文件
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"dialog_{date}.json"
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for entry in data:
                            ts = entry['timestamp']
                            self.memory_index.append(ts)
                            self.memory_data[ts] = entry
                            count += 1
                except Exception as e:
                    print(f"[记忆系统] 加载文件失败 {file_path}: {e}")
        
        # 排序索引
        self.memory_index.sort()
        print(f"[记忆系统] 加载完成,共 {count} 条记忆")
    
    def _load_core_memories(self):
        """加载核心记忆"""
        core_file = self.memory_dir / "core_memories.json"
        if core_file.exists():
            try:
                with open(core_file, 'r', encoding='utf-8') as f:
                    self.core_memories = json.load(f)
                print(f"[记忆系统] 加载核心记忆: {len(self.core_memories)} 条")
            except Exception as e:
                print(f"[记忆系统] 加载核心记忆失败: {e}")
    
    def add_conversation(
        self,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict] = None
    ):
        """
        添加一轮对话到记忆
        
        Args:
            user_message: 用户消息
            assistant_message: 助手回复
            metadata: 额外元数据 (如情绪、音色等)
        """
        timestamp = time.time()
        date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        
        entry = {
            'timestamp': timestamp,
            'date': date,
            'user': user_message,
            'assistant': assistant_message,
            'metadata': metadata or {},
        }
        
        # 添加到索引
        self.memory_index.append(timestamp)
        self.memory_data[timestamp] = entry
        
        # 保存到文件
        self._save_to_file(date, entry)
        
        print(f"[记忆系统] 保存对话: {user_message[:20]}...")
    
    def _save_to_file(self, date: str, entry: Dict):
        """保存对话到日期文件"""
        file_path = self.memory_dir / f"dialog_{date}.json"
        
        # 读取现有数据
        data = []
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                pass
        
        # 添加新条目
        data.append(entry)
        
        # 写回文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[记忆系统] 保存失败: {e}")
    
    def search_by_time_range(
        self,
        start_time: float,
        end_time: float,
        limit: int = 10
    ) -> List[Dict]:
        """
        在时间范围内搜索对话 (借鉴 MoeChat 的二分查找)
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            limit: 最多返回数量
        
        Returns:
            对话列表
        """
        # 使用二分查找定位范围
        start_idx = bisect_left(self.memory_index, start_time)
        end_idx = bisect_right(self.memory_index, end_time)
        
        if start_idx >= len(self.memory_index) or end_idx == 0:
            return []
        
        # 提取范围内的记忆
        results = []
        for idx in range(start_idx, min(end_idx, start_idx + limit)):
            ts = self.memory_index[idx]
            results.append(self.memory_data[ts])
        
        return results
    
    def search_recent(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """搜索最近 N 小时的对话"""
        now = time.time()
        start_time = now - hours * 3600
        return self.search_by_time_range(start_time, now, limit)
    
    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        按关键词搜索对话
        
        实际应用可结合 Embedding 向量相似度
        """
        results = []
        
        for ts in reversed(self.memory_index):  # 从新到旧
            entry = self.memory_data[ts]
            
            # 简单关键词匹配
            if (keyword.lower() in entry['user'].lower() or
                keyword.lower() in entry['assistant'].lower()):
                results.append(entry)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def add_core_memory(self, memory: str):
        """添加核心记忆"""
        if memory not in self.core_memories:
            self.core_memories.append(memory)
            self._save_core_memories()
            print(f"[记忆系统] 添加核心记忆: {memory}")
    
    def _save_core_memories(self):
        """保存核心记忆"""
        core_file = self.memory_dir / "core_memories.json"
        try:
            with open(core_file, 'w', encoding='utf-8') as f:
                json.dump(self.core_memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[记忆系统] 保存核心记忆失败: {e}")
    
    def get_context_for_llm(
        self,
        user_message: str,
        recent_count: int = 5
    ) -> str:
        """
        为 LLM 构建上下文 (结合核心记忆 + 最近对话)
        
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 1. 核心记忆
        if self.core_memories:
            context_parts.append("【核心记忆】")
            for mem in self.core_memories[:5]:  # 最多5条
                context_parts.append(f"- {mem}")
            context_parts.append("")
        
        # 2. 最近对话
        recent = self.search_recent(hours=24, limit=recent_count)
        if recent:
            context_parts.append("【最近对话】")
            for conv in recent[-recent_count:]:
                time_str = datetime.fromtimestamp(conv['timestamp']).strftime("%H:%M")
                context_parts.append(f"[{time_str}] 用户: {conv['user']}")
                context_parts.append(f"[{time_str}] 助手: {conv['assistant']}")
            context_parts.append("")
        
        return "\n".join(context_parts)


# 使用示例
if __name__ == "__main__":
    memory = SimpleMemorySystem("data/test_memories")
    
    # 添加对话
    memory.add_conversation(
        user_message="今天天气怎么样?",
        assistant_message="今天晴天,温度25度,适合出门。",
        metadata={'emotion': 'neutral'}
    )
    
    # 搜索最近对话
    recent = memory.search_recent(hours=24)
    print(f"\n最近对话: {len(recent)} 条")
    
    # 添加核心记忆
    memory.add_core_memory("用户喜欢早上喝咖啡")
    
    # 获取 LLM 上下文
    context = memory.get_context_for_llm("早上好")
    print(f"\nLLM 上下文:\n{context}")
