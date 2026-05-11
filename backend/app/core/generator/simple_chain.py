from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.core.generator.base import BaseGenerator
from app.config import settings

RAG_PROMPT_TEMPLATE = """你是一个专业的劳动法律师助手。你必须严格基于以下提供的法律条文和参考资料来回答问题。

【重要规则】
1. 回答必须严格基于提供的参考资料，禁止编造任何法律条文或内容
2. 如果参考资料中没有相关信息，必须明确告知"根据现有资料无法回答该问题"
3. 引用法律条文时，必须标明出处（如：《劳动合同法》第X条），且该条文必须出现在参考资料中
4. 不要添加参考资料之外的法律知识或个人理解
5. 对不确定的内容使用"根据参考资料""可能"等限定词

参考资料：
{context}

用户问题：{question}

请严格基于以上参考资料回答用户的问题："""


class SimpleChainGenerator(BaseGenerator):
    """V1: 基于LCEL的简单链式生成器。"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(
        self,
        question: str,
        context_docs: List[Tuple[Document, float]],
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
        })
        return answer
