from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.core.generator.base import BaseGenerator
from app.config import settings
from app.core.generator.prompts import ANSWER_STYLE_RULES
from app.core.generator.guardrails import AnswerGuardrails

RAG_PROMPT_TEMPLATE = """{style_rules}

参考资料：
{context}

用户问题：{question}

请基于以上参考资料回答用户的问题："""


class SimpleChainGenerator(BaseGenerator):
    """V1: 基于LCEL的简单链式生成器。"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            extra_body={"enable_thinking": False},
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(
        self,
        question: str,
        context_docs: List[Tuple[Document, float]],
        skip_guardrails: bool = False,
    ) -> str:
        context = "\n\n".join(
            [
                f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
                for doc, score in context_docs
            ]
        ) if context_docs else "未找到相关参考资料。"

        answer = self.chain.invoke({
            "context": context,
            "question": question,
            "style_rules": ANSWER_STYLE_RULES,
        })

        # 回答后校验：零 token 消耗，代码层面追加风险提示
        # 评估模式下跳过护栏追加，避免RAGAS将追加内容判为幻觉
        if not skip_guardrails:
            answer = AnswerGuardrails.check(answer)
        return answer
