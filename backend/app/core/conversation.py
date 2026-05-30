"""V3: 对话历史管理器。

支持多轮对话，维护每个会话的对话历史，
并在检索时将上下文融入查询改写。
持久化到JSON文件，后端重启不丢失。
"""
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from app.models.schemas import ChatMessage, MessageRole
from app.config import settings


class ConversationManager:
    """对话历史管理器，按conversation_id管理多轮对话，支持JSON持久化。"""

    def __init__(self, max_history: int = 20):
        # conversation_id -> List[ChatMessage]
        self._conversations: Dict[str, List[ChatMessage]] = {}
        self.max_history = max_history
        self._data_dir = Path(settings.CONVERSATION_DATA_PATH)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _conv_path(self, conversation_id: str) -> Path:
        """获取单个对话的JSON文件路径。"""
        # 用conversation_id的hash前缀作为文件名，避免特殊字符
        safe_id = conversation_id.replace("/", "_").replace("\\", "_")
        return self._data_dir / f"{safe_id}.json"

    def _load_conversation(self, conversation_id: str) -> List[ChatMessage]:
        """从磁盘加载对话历史（如果尚未加载）。"""
        if conversation_id in self._conversations:
            return self._conversations[conversation_id]

        path = self._conv_path(conversation_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                messages = [
                    ChatMessage(
                        role=MessageRole(msg["role"]),
                        content=msg["content"],
                        timestamp=datetime.fromisoformat(msg["timestamp"]) if msg.get("timestamp") else None,
                    )
                    for msg in data
                ]
                self._conversations[conversation_id] = messages
                return messages
            except Exception:
                self._conversations[conversation_id] = []
                return []
        else:
            self._conversations[conversation_id] = []
            return []

    def _save_conversation(self, conversation_id: str):
        """将对话历史持久化到磁盘。"""
        messages = self._conversations.get(conversation_id, [])
        path = self._conv_path(conversation_id)
        data = [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            for msg in messages
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_message(self, conversation_id: str, role: MessageRole, content: str):
        """添加一条消息到对话历史，并持久化。"""
        if conversation_id not in self._conversations:
            self._load_conversation(conversation_id)

        self._conversations[conversation_id].append(
            ChatMessage(role=role, content=content, timestamp=datetime.now())
        )

        # 保留最近N条
        if len(self._conversations[conversation_id]) > self.max_history:
            self._conversations[conversation_id] = self._conversations[conversation_id][-self.max_history:]

        self._save_conversation(conversation_id)

    def get_history(self, conversation_id: str) -> List[ChatMessage]:
        """获取指定会话的对话历史。"""
        return self._load_conversation(conversation_id)

    def build_context_string(self, conversation_id: str, last_n: int = 6) -> str:
        """构建对话上下文字符串，用于Prompt拼接。

        Args:
            conversation_id: 会话ID
            last_n: 取最近N轮对话（1轮=1问1答，所以取last_n*2条消息）
        """
        history = self._load_conversation(conversation_id)
        if not history:
            return ""

        # 取最近N条
        recent = history[-(last_n * 2):]
        lines = []
        for msg in recent:
            if msg.role == MessageRole.USER:
                lines.append(f"用户: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                lines.append(f"助手: {msg.content}")
        return "\n".join(lines)

    def clear(self, conversation_id: str):
        """清除指定会话的历史（同时删除磁盘文件）。"""
        self._conversations.pop(conversation_id, None)
        path = self._conv_path(conversation_id)
        if path.exists():
            path.unlink()

    def has_history(self, conversation_id: str) -> bool:
        """是否有对话历史。"""
        return len(self._load_conversation(conversation_id)) > 0


# 全局单例
conversation_manager = ConversationManager()
