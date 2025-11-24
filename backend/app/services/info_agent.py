"""LangChain 智能体封装：用于需要联网或文件工具的查询"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.services import history_pg

try:  # 按需加载 LangChain
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langchain_community.tools import (
        DuckDuckGoSearchRun,
        RequestsGetTool,
        RequestsPostTool,
        WikipediaQueryRun,
        ArxivQueryRun,
    )
    from langchain_community.tools.file_management import ListDirectoryTool, ReadFileTool
    from langchain_community.utilities.requests import RequestsWrapper
    from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
    from langchain_core.tools import tool

    # Python REPL 工具
    try:
        from langchain_experimental.tools.python.tool import PythonREPLTool

        PYTHON_REPL_AVAILABLE = True
    except ImportError:
        PYTHON_REPL_AVAILABLE = False

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False
    PYTHON_REPL_AVAILABLE = False

logger = logging.getLogger(__name__)
settings = get_settings()


class InfoAgent:
	"""基于 LangChain ReAct 的查询 Agent。"""

	def __init__(self) -> None:
		if not LANGCHAIN_AVAILABLE:
			raise RuntimeError("LangChain 未安装，无法启用 Agent")

		# 初始化工具列表
		self.tools = []

		# 1. 时间相关工具（自定义）
		@tool("get_current_datetime", return_direct=False)
		def _get_current_datetime(format: str = "full") -> str:
			"""
			获取当前日期和时间。

			Args:
				format: 返回格式，可选值：
					- "full": 完整日期时间，如 "2025-11-24 16:30:45 星期日"
					- "date": 仅日期，如 "2025-11-24"
					- "time": 仅时间，如 "16:30:45"
					- "datetime": 日期时间，如 "2025-11-24 16:30:45"
					- "chinese": 中文格式，如 "2025年11月24日 16时30分45秒 星期日"

			Returns:
				格式化的日期时间字符串

			Examples:
				- 用户问"今天几号"、"现在几点"、"今天星期几"时调用此工具
				- 用户问"今天日期"、"当前时间"时调用此工具
			"""
			now = datetime.now()
			weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
			weekday = weekday_cn[now.weekday()]

			if format == "date":
				return now.strftime("%Y-%m-%d")
			elif format == "time":
				return now.strftime("%H:%M:%S")
			elif format == "datetime":
				return now.strftime("%Y-%m-%d %H:%M:%S")
			elif format == "chinese":
				return f"{now.year}年{now.month}月{now.day}日 {now.hour}时{now.minute}分{now.second}秒 {weekday}"
			else:  # full
				return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {weekday}"

		self.tools.append(_get_current_datetime)

		# 2. 网络搜索工具
		self.tools.append(DuckDuckGoSearchRun())

		# 3. HTTP 请求工具
		requests_wrapper = RequestsWrapper()
		self.tools.append(RequestsGetTool(
			requests_wrapper=requests_wrapper,
			allow_dangerous_requests=True
		))
		self.tools.append(RequestsPostTool(
			requests_wrapper=requests_wrapper,
			allow_dangerous_requests=True
		))

		# 4. 知识库工具
		self.tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
		self.tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper()))

		# 5. 文件操作工具
		self.tools.append(ListDirectoryTool())
		self.tools.append(ReadFileTool())

		# 6. Python REPL 工具（如果可用）
		if PYTHON_REPL_AVAILABLE:
			self.tools.append(PythonREPLTool())
		if settings.history_db_enabled:
			@tool("search_conversation_history", return_direct=False)
			def _search_conversation_history(q: str) -> str:
				"""
				查询最近30天的历史对话记录。
				当用户询问过去的对话/之前的问题时调用；若无特定关键词可传空字符串，
				工具将返回最近的若干对话，模型再自行筛选关联内容。
				字段: session_id, conversation_date(YYYY-MM-DD), created_at(ISO), user_message, assistant_message, images(JSON)。
				时间相关问题可结合 created_at 理解最近时间。更精确的时间窗口可用 run_history_sql。
				"""
				query = (q or "").strip() or None
				return history_pg.format_history_rows(
					history_pg.search_history(query=query, days=30, limit=12)
				)

			self.tools.append(_search_conversation_history)
			@tool("run_history_sql", return_direct=False)
			def _run_history_sql(sql: str) -> str:
				"""
				执行只读 SQL 查询历史记录。仅支持 SELECT/WITH，可用 {table} 占位表名。
				表结构: session_id TEXT, conversation_date TEXT(YYYY-MM-DD), created_at TEXT(ISO), user_message TEXT,
				assistant_message TEXT, images TEXT(JSON)。
				时间查询示例:
				  SELECT user_message,assistant_message,created_at FROM {table}
				  WHERE datetime(created_at) BETWEEN '2025-11-24 08:00:00' AND '2025-11-24 09:00:00'
				  ORDER BY datetime(created_at) DESC LIMIT 5;
				"""
				return history_pg.run_readonly_sql(sql, max_rows=20)

			self.tools.append(_run_history_sql)
		self.trace_enabled = settings.agent_trace_enabled
		self.recursion_limit = settings.agent_recursion_limit

		system_prompt = (
			"你是一个智能助手。你拥有多种工具，遇到需要实时数据、计算、搜索的问题时，必须优先调用工具获取准确答案。\n\n"

			"# 核心原则\n"
			"1. **永远不要说\"我无法获取\"或\"我不能提供\"** — 你有工具可以用！\n"
			"2. **先思考需要什么信息，再选择工具** — 仔细阅读工具描述\n"
			"3. **工具失败才说做不到** — 不要提前放弃\n"
			"4. **一个工具不行就换另一个** — 多尝试几种方式\n\n"

			"# 常见场景与必须使用的工具\n"

			"## 🌤️ 实时信息类（天气、新闻、股价等）\n"
			"- 用户问\"今天天气\"、\"北京天气\"、\"明天下雨吗\" → **必须**调用 `duckduckgo_search` 搜索最新天气\n"
			"- 用户问\"最新新闻\"、\"今天发生了什么\" → **必须**调用 `duckduckgo_search` 搜索新闻\n"
			"- ⚠️ 不要说\"我无法获取天气\"，你有搜索工具可以查！\n\n"

			"## 📅 时间日期类\n"
			"- 用户问\"今天几号\"、\"现在几点\"、\"今天星期几\" → **必须**调用 `get_current_datetime` 工具\n"
			"- ⚠️ 不要凭记忆回答时间，必须用工具！\n\n"

			"## 🧮 数学计算类\n"
			"- 用户问\"123乘以456\"、\"计算平方根\"、\"多少等于\" → **必须**调用 `Python_REPL` 工具执行计算\n"
			"- ⚠️ 注意：`requests_get`/`requests_post` 是访问网页的，不是做计算的！\n\n"

			"## 📚 知识检索类\n"
			"- 用户问\"搜索...\"、\"查一下...\" → 优先使用 `duckduckgo_search`\n"
			"- 用户问百科知识（如\"什么是量子计算\"） → 使用 `wikipedia` 工具\n"
			"- 用户问学术论文 → 使用 `arxiv` 工具\n\n"

			"## 💬 历史对话类\n"
			"- 用户问\"我之前说了什么\"、\"我们聊过什么\" → 使用 `search_conversation_history` 工具\n"
			"- 需要精确时间范围查询 → 使用 `run_history_sql` 工具\n\n"

			"## 🌐 HTTP 请求类\n"
			"- 需要访问特定网页或API → 使用 `requests_get` 或 `requests_post` 工具\n"
			"- ⚠️ 这些工具只用于网络请求，不能用于计算或文件操作！\n\n"

			"## 📁 文件操作类\n"
			"- 查看目录内容 → 使用 `list_directory` 工具\n"
			"- 读取文件内容 → 使用 `read_file` 工具\n\n"

			"# 错误示例（禁止这样做）\n"
			"❌ 用户:\"今天北京天气怎么样？\" 你回答:\"我无法获取实时天气\" — 错误！应该用 duckduckgo_search 搜索\n"
			"❌ 用户:\"12345*67890等于多少？\" 你调用 requests_post — 错误！应该用 Python_REPL 计算\n"
			"❌ 用户:\"今天几号？\" 你回答:\"11月24日\" — 错误！必须调用 get_current_datetime 工具\n\n"

			"# 正确示例\n"
			"✅ 用户:\"今天北京天气怎么样？\" → 调用 duckduckgo_search(\"北京今天天气\")\n"
			"✅ 用户:\"12345*67890等于多少？\" → 调用 Python_REPL(\"12345 * 67890\")\n"
			"✅ 用户:\"今天几号？\" → 调用 get_current_datetime(format=\"full\")\n\n"

			"# 工具使用流程\n"
			"1. 识别用户需求类型（实时信息/计算/时间/搜索等）\n"
			"2. 根据上面的场景选择对应工具\n"
			"3. 调用工具获取结果\n"
			"4. 用自然、口语化的中文总结结果\n"
			"5. 如果工具失败，尝试其他工具，实在不行再告诉用户无法完成\n\n"

			"记住：你有工具，不要轻易说做不到！先用工具试试再说。"
		)

		llm = ChatOpenAI(
			api_key=settings.dashscope_api_key,
			base_url=str(settings.qwen_base_url),
			model=settings.default_llm_model,
			temperature=0.2,
		)

		self.agent_graph = create_agent(llm, self.tools, system_prompt=system_prompt)

		# 输出工具列表
		tool_names = [t.name if hasattr(t, 'name') else str(t) for t in self.tools]
		logger.info(f"[Agent] LangChain agent graph 已初始化，共 {len(self.tools)} 个工具")
		logger.info(f"[Agent] 可用工具列表: {', '.join(tool_names)}")

	async def arun(self, history: List[Dict[str, str]], user_text: str) -> str:
		messages = []
		for turn in history[-6:]:
			role = turn.get("role")
			content = turn.get("content", "")
			if not content:
				continue
			if role in ("user", "assistant"):
				messages.append({"role": role, "content": content})

		messages.append({"role": "user", "content": user_text})

		payload = {"messages": messages}

		logger.info(f"[Agent] 📨 用户输入: {user_text}")
		logger.info(f"[Agent] 📚 历史消息: {len(messages)-1} 条")

		# 始终使用流式处理以便记录详细日志
		loop = asyncio.get_running_loop()
		return await loop.run_in_executor(None, self._stream_and_collect, payload)

	def _stream_and_collect(self, payload: Dict[str, Any]) -> str:
		final_output: Optional[str] = None
		step_count = 0

		logger.info("[Agent] 🔄 开始流式处理...")

		for chunk in self.agent_graph.stream(payload, config={"recursion_limit": self.recursion_limit}):
			step_count += 1

			for node_name, state_update in chunk.items():
				logger.info(f"[Agent] 📍 Step {step_count}: Node={node_name}")

				if isinstance(state_update, dict):
					messages = state_update.get("messages", [])
					if messages:
						last_msg = messages[-1]

						# 提取消息类型和内容
						msg_type = None
						content = None
						tool_calls = None

						if isinstance(last_msg, dict):
							msg_type = last_msg.get("type", "unknown")
							content = last_msg.get("content", "")
							tool_calls = last_msg.get("tool_calls")
						elif hasattr(last_msg, "type"):
							msg_type = getattr(last_msg, "type", "unknown")
							content = getattr(last_msg, "content", "")
							tool_calls = getattr(last_msg, "tool_calls", None) if hasattr(last_msg, "tool_calls") else None

						# 记录详细信息
						logger.info(f"[Agent]   消息类型: {msg_type}")

						# 如果是工具调用
						if tool_calls:
							for tool_call in tool_calls:
								if isinstance(tool_call, dict):
									tool_name = tool_call.get("name", "unknown")
									tool_args = tool_call.get("args", {})
								elif hasattr(tool_call, "name"):
									tool_name = getattr(tool_call, "name", "unknown")
									tool_args = getattr(tool_call, "args", {})
								else:
									tool_name = str(tool_call)
									tool_args = {}

								logger.info(f"[Agent]   🔧 调用工具: {tool_name}")
								logger.info(f"[Agent]   📝 参数: {tool_args}")

						# 记录内容
						if content and isinstance(content, str):
							if len(content) > 200:
								logger.info(f"[Agent]   💬 内容: {content[:200]}...")
							else:
								logger.info(f"[Agent]   💬 内容: {content}")

							# 如果是 AI 的最终回复，保存
							if msg_type in ("ai", "AIMessage") and not tool_calls:
								final_output = content
				else:
					logger.debug(f"[Agent]   状态更新类型: {type(state_update)}")

		logger.info(f"[Agent] ✅ 流式处理完成，共 {step_count} 步")

		if final_output is None:
			logger.warning("[Agent] ⚠️  未从流中获取到输出，尝试完整调用...")
			result = self.agent_graph.invoke(payload, config={"recursion_limit": self.recursion_limit})
			final_output = self._extract_output(result)

		return final_output

	def _extract_output(self, result: Any) -> str:
		if isinstance(result, dict):
			messages = result.get("messages", [])
			if messages:
				last_msg = messages[-1]
				if isinstance(last_msg, dict):
					content = last_msg.get("content")
					if isinstance(content, str):
						return content
				elif hasattr(last_msg, "content"):
					content = getattr(last_msg, "content")
					if isinstance(content, str):
						return content
		return str(result)


_info_agent: Optional[InfoAgent] = None


def get_info_agent() -> Optional[InfoAgent]:
	global _info_agent
	if not settings.agent_enabled:
		return None
	if not LANGCHAIN_AVAILABLE:
		logger.warning("[Agent] LangChain 未安装，智能体已禁用")
		return None
	if _info_agent is None:
		_info_agent = InfoAgent()
	return _info_agent
