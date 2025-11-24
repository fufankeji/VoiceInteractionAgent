"""SQLite 单文件对话历史存取 & LangChain Tool 支持"""
from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from typing import List, Tuple
from urllib.parse import urlparse

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _get_conn():
    """获取 SQLite 连接并建表。history_db_url 应为 sqlite:///path/to.db"""
    if not settings.history_db_enabled or not settings.history_db_url:
        raise RuntimeError("历史记录数据库未启用或 URL 未配置")

    url = settings.history_db_url
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if not scheme.startswith("sqlite"):
        raise RuntimeError("仅支持 sqlite，示例: sqlite:///./data/history.db")

    db_path = url.replace("sqlite:///", "")
    from pathlib import Path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_table(conn)
    return conn


def _ensure_table(conn):
    table = settings.history_table_name
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        conversation_date TEXT NOT NULL,
        user_message TEXT NOT NULL,
        assistant_message TEXT NOT NULL,
        images TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table}(conversation_date);
    CREATE INDEX IF NOT EXISTS idx_{table}_created_at ON {table}(created_at);
    """
    with conn:
        conn.executescript(ddl)


def save_dialog(
    session_id: str,
    user_message: str,
    assistant_message: str,
    images: List[str] | None = None,
) -> None:
    """保存一轮对话到 SQLite（按天分桶）"""
    if not settings.history_db_enabled:
        return
    try:
        conn = _get_conn()
    except Exception as exc:  # pragma: no cover
        logger.debug("[HistoryDB] skip save: %s", exc)
        return

    table = settings.history_table_name
    today = datetime.date.today().isoformat()
    images_json = json.dumps(images or None, ensure_ascii=False)
    with conn:
        conn.execute(
            f"""INSERT INTO {table}
            (session_id, conversation_date, user_message, assistant_message, images)
            VALUES (?, ?, ?, ?, ?)""",
            (session_id, today, user_message, assistant_message, images_json),
        )
    conn.close()


def search_history(
    query: str | None = None,
    days: int = 7,
    limit: int = 5,
) -> List[Tuple[str, str, str, str]]:
    """
    查询最近 N 天历史，支持可选关键词。
    query 为空则返回最近 limit 条，供 Agent 自主回溯。
    """
    if not settings.history_db_enabled:
        return []
    try:
        conn = _get_conn()
    except Exception as exc:  # pragma: no cover
        logger.debug("[HistoryDB] skip search: %s", exc)
        return []

    table = settings.history_table_name
    with conn:
        if query:
            cur = conn.execute(
                f"""
                SELECT conversation_date, created_at, user_message, assistant_message
                FROM {table}
                WHERE datetime(created_at) >= datetime('now', ?)
                  AND (user_message LIKE ? OR assistant_message LIKE ?)
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (f"-{days} days", f"%{query}%", f"%{query}%", limit),
            )
        else:
            cur = conn.execute(
                f"""
                SELECT conversation_date, created_at, user_message, assistant_message
                FROM {table}
                WHERE datetime(created_at) >= datetime('now', ?)
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (f"-{days} days", limit),
            )
        rows = cur.fetchall()
    conn.close()
    return [
        (row[0], row[1], row[2], row[3])
        for row in rows
    ]


def format_history_rows(rows: List[Tuple[str, str, str, str]]) -> str:
    """将查询结果格式化为可读文本，供 LLM 消化"""
    if not rows:
        return "未找到相关历史记录。"
    lines = []
    for date_str, ts, user_msg, assistant_msg in rows:
        lines.append(
            f"[{date_str} {ts}] 用户: {user_msg}\n助手: {assistant_msg}"
        )
    return "\n\n".join(lines)


# === 允许 Agent 查看 schema / 执行只读查询 ===
SCHEMA_DESCRIPTION = """表名: {table}
列:
- id INTEGER (自增主键)
- session_id TEXT
- conversation_date TEXT (YYYY-MM-DD)
- user_message TEXT
- assistant_message TEXT
- images TEXT (JSON)
- created_at TEXT (ISO 时间戳)
索引: conversation_date, created_at
建议: 使用 SELECT/WHERE/ORDER BY/LIMIT，禁止写操作。""".format(table=settings.history_table_name)


def get_schema_description() -> str:
    """返回当前历史表的 schema 描述，供 LLM 参考"""
    return SCHEMA_DESCRIPTION


def run_readonly_sql(sql: str, max_rows: int = 20) -> str:
    """
    执行只读 SQL（仅允许 SELECT/WITH），返回文本结果。
    由 Agent 生成查询语句时调用，以提升灵活性。
    """
    if not settings.history_db_enabled:
        return "历史库未启用。"

    sanitized = (sql or "").strip().rstrip(";")
    low = sanitized.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "仅支持只读查询（SELECT/WITH）。"

    try:
        conn = _get_conn()
    except Exception as exc:  # pragma: no cover
        return f"无法连接数据库: {exc}"

    table = settings.history_table_name
    # 防止恶意跨表：简单替换占位符 {table}
    sanitized = sanitized.replace("{table}", table)

    try:
        with conn:
            cur = conn.execute(sanitized)
            rows = cur.fetchmany(max_rows)
            cols = [desc[0] for desc in cur.description] if cur.description else []
    except Exception as exc:
        conn.close()
        return f"查询失败: {exc}"

    conn.close()

    if not rows:
        return "无结果。"

    # 简单格式化表格
    col_line = " | ".join(cols)
    data_lines = []
    for row in rows:
        data_lines.append(" | ".join(str(x) for x in row))
    return col_line + "\n" + "\n".join(data_lines)


def fetch_recent_dialogs(
    session_id: str | None,
    limit_pairs: int = 10,
    include_all_sessions: bool = False,
) -> List[dict]:
    """
    获取指定 session 的最近若干轮对话，返回 OpenAI 格式的 messages 列表（按时间正序）。
    每轮包含 user 和 assistant 两条消息，因此最大条数约为 2*limit_pairs。
    当 include_all_sessions=True 时，不按 session_id 过滤（全局合并）。
    """
    if not settings.history_db_enabled:
        return []
    try:
        conn = _get_conn()
    except Exception as exc:  # pragma: no cover
        logger.debug("[HistoryDB] skip fetch: %s", exc)
        return []

    table = settings.history_table_name
    with conn:
        if include_all_sessions or not session_id:
            cur = conn.execute(
                f"""
                SELECT session_id, conversation_date, created_at, user_message, assistant_message, images
                FROM {table}
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit_pairs,),
            )
        else:
            cur = conn.execute(
                f"""
                SELECT session_id, conversation_date, created_at, user_message, assistant_message, images
                FROM {table}
                WHERE session_id = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (session_id, limit_pairs),
            )
        rows = cur.fetchall()
    conn.close()

    # 最新在前，转换为时间正序
    rows = list(reversed(rows))
    messages: List[dict] = []
    for row in rows:
        _session, _date, _ts, user_msg, assistant_msg, images_json = row
        # 如果有历史图片，追加提示文本，便于模型意识到有图片出现
        if images_json:
            try:
                imgs = json.loads(images_json)
                if imgs:
                    preview = ""
                    if isinstance(imgs, list):
                        preview = ", ".join(str(i)[:60] for i in imgs[:2])
                        if len(imgs) > 2:
                            preview += " 等"
                    messages.append({
                        "role": "user",
                        "content": f"{user_msg}\n[历史图片 {len(imgs)} 张]: {preview}"
                    })
                else:
                    messages.append({"role": "user", "content": user_msg})
            except Exception:
                messages.append({"role": "user", "content": user_msg})
        else:
            messages.append({"role": "user", "content": user_msg})

        messages.append({"role": "assistant", "content": assistant_msg})
    # 额外截断到 2*limit_pairs 防止溢出
    return messages[-2 * limit_pairs :]
