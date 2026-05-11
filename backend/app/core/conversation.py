"""V3: 对话历史管理器。

支持多轮对话，维护每个会话的对话历史，
并在检索时将上下文融入查询改写。
"""
from typing import List, Dict, Optional
from datetime import datetime
from app.models.schemas import ChatMessage, MessageRole


class ConversationManager:
    """对话历史管理器，按conversation_id管理多轮对话。"""

    def __init__(self, max_history: int = 20):
        # conversation_id -> List[ChatMessage]
        self._conversations: Dict[str, List[ChatMessage]] = {}
        self.max_history = max_history

    def add_message(self, conversation_id: str, role: MessageRole, content: str):
        """添加一条消息到对话历史。"""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        self._conversations[conversation_id].append(
            ChatMessage(role=role, content=content, timestamp=datetime.now())
        )

        # 保留最近N条
        if len(self._conversations[conversation_id]) > self.max_history:
            self._conversations[conversation_id] = self._conversations[conversation_id][-self.max_history:]

    def get_history(self, conversation_id: str) -> List[ChatMessage]:
        """获取指定会话的对话历史。"""
        return self._conversations.get(conversation_id, [])

    def build_context_string(self, conversation_id: str, last_n: int = 6) -> str:
        """构建对话上下文字符串，用于Prompt拼接。

        Args:
            conversation_id: 会话ID
            last_n: 取最近N轮对话（1轮=1问1答，所以取last_n*2条消息）
        """
        history = self._conversations.get(conversation_id, [])
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
        """清除指定会话的历史。"""
        self._conversations.pop(conversation_id, None)

    def has_history(self, conversation_id: str) -> bool:
        """是否有对话历史。"""
        return len(self._conversations.get(conversation_id, [])) > 0


# 全局单例
conversation_manager = ConversationManager()
