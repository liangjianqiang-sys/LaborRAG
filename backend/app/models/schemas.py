from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户提问")
    conversation_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    history: Optional[list[ChatMessage]] = Field(None, description="对话历史")
    skip_guardrails: bool = Field(False, description="跳过护栏追加（评估模式使用，避免RAGAS幻觉误判）")


class SourceDocument(BaseModel):
    content: str = Field(..., description="文档内容")
    source: str = Field(..., description="文档来源")
    score: float = Field(..., description="相关度分数")
    page: Optional[int] = Field(None, description="页码")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI回答")
    sources: list[SourceDocument] = Field(default_factory=list, description="参考来源")
    conversation_id: str = Field(..., description="会话ID")
    rag_mode: str = Field("simple", description="RAG模式: simple | crag | agent")
    crag_steps: list[str] = Field(default_factory=list, description="CRAG工作流步骤记录")
    rewritten_question: str = Field("", description="改写后的问题（CRAG模式）")
    confidence: float = Field(0.0, description="回答置信度(0-1)，低于0.5时建议咨询律师")
    disclaimer: bool = Field(True, description="是否已追加免责声明")
    full_contexts: list[str] = Field(default_factory=list, description="完整检索文档内容（供评估使用，不截断）")


class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    total_chunks: int
    vector_store_exists: bool
    embedding_model: str
    llm_model: str


class UploadResponse(BaseModel):
    success: bool
    filename: str
    chunk_count: int
    message: str


class BuildRequest(BaseModel):
    rebuild: bool = Field(False, description="是否重建向量库")
