# 忠实度(Faithfulness)提升计划

> 当前得分：~0.61 | 目标：≥0.80 | 2026-05-30

---

## 根因诊断

忠实度卡在 40-60% 是**多因素叠加**的结果，不是单一问题。按影响从大到小：

| # | 根因 | 当前影响 | 修复难度 |
|---|------|----------|----------|
| 1 | **护栏(Guardrails)追加的"注意事项"不在检索上下文中** | 高 — 约40%的回答被追加了上下文不存在的法律提示 | ⭐ 简单 |
| 2 | **检索召回率太低 (recall@5=0.61)** | 高 — 约40%的相关法条没有被检索到 | ⭐⭐ 中等 |
| 3 | **Prompt 要求解释法条适用前提，但前提法条可能不在上下文中** | 中 — Rule 12 强制引用前提法条 | ⭐⭐ 中等 |
| 4 | **计算器硬编码的法条引用不在评估上下文中** | 中 — 计算类问题忠实度仅 0.54 | ⭐⭐ 中等 |
| 5 | **Validator 明确允许引用上下文之外的法条** | 中 — 验证节点放行"幻觉" | ⭐ 简单 |
| 6 | **上下文压缩器阈值(0.5)可能丢弃相关法条** | 中 | ⭐ 简单 |
| 7 | **CRAG 内部忠实度评估器只读 300 字符** | 低 | ⭐ 简单 |
| 8 | **LLM 的"乐于助人"倾向 vs 拒绝回答** | 低-中 | ⭐⭐⭐ 困难 |

---

## 修复计划

### 第一阶段：快速止血（预计提升 10-15 分）

#### 1.1 修复护栏在评估时追加额外内容
**文件：** `backend/app/core/generator/guardrails.py`

问题：`skip_guardrails=True` 只在 V4 Agent 模式生效，V1/V2/V3 评估时护栏仍然追加"注意事项"，这些内容不在上下文中，RAGAS 忠实度直接判为幻觉。

修改方案：
```
方案B（推荐，V4已实现）：确保所有 RAG 模式的生成链路都尊重 skip_guardrails 参数。
  - V4 Agent 已支持 skip_guardrails（agent_graph.py:924）
  - V1/V2 SimpleChain 和 V3 CRAG 仍无条件调用 AnswerGuardrails.check()
  - 需要给 simple_chain.py 和 crag_graph.py 的 generate/run 方法添加 skip_guardrails 参数
  - rag_engine.py 在评估模式下统一传递 skip_guardrails=True

方案A（备选）：guardrails.py 中 CHECK_RULES 的 hint 文本追加到
  ChatResponse.disclaimer_note 字段，而不是直接拼接到 answer。
  这样前端可以展示提示，但不影响忠实度评分。
```

#### 1.2 降低上下文压缩器阈值
**文件：** `backend/app/core/retriever/compressor.py`

将 `COMPRESSION_THRESHOLD` 从 `0.5` 降到 `0.35`，让更多相关法条通过压缩器进入最终的上下文窗口。

#### 1.3 提升 CRAG 内部评估器的截断长度
**文件：** `backend/app/core/generator/crag_graph.py`

```python
GRADE_ANSWER_CONTEXT_CHARS = 300  → 800
GRADE_ANSWER_ANSWER_CHARS = 600   → 1200
```

---

### 第二阶段：检索增强（预计提升 10-15 分）

#### 2.1 提升 TOP_K 和降低 SCORE_THRESHOLD
**文件：** `backend/.env`

```
TOP_K=8  → TOP_K=12              # 检索更多候选
SCORE_THRESHOLD=0.2 → 0.15       # 放宽阈值，提高召回（当前默认已从0.3统一为0.2）
RERANK_TOP_K=5 → RERANK_TOP_K=8  # 重排序后保留更多（当前默认已从3提升为5）
```

#### 2.2 查询增强优化
**文件：** `backend/app/core/retriever/query_enhance.py`

当前已有口语→法言法语映射，可以扩充：
- 增加同义词扩展（如 "辞退" → "解除劳动合同 辞退 开除"）
- 增加法条编号识别（如用户说"第38条"→ 自动补充法律名称）

#### 2.3 混合检索权重调整
**文件：** `backend/.env`

```
VECTOR_WEIGHT=0.5 → 0.45         # 降低向量权重（当前默认已为0.5）
BM25_WEIGHT=0.5 → 0.55           # 提高BM25权重（关键词对法条编号更敏感）
```

---

### 第三阶段：Prompt 优化（预计提升 5-8 分）

#### 3.1 解决 Rule 12 与 Rule 1/5 的矛盾
**文件：** `backend/app/core/generator/prompts.py`

当前矛盾：Rule 12 要求"必须说明适用前提"，但前提法条可能不在检索上下文中，违反 Rule 1/5。

修改方案：
```
原 Rule 12：
  "对于引用法条的回答，必须说明该法条的适用前提"
  → 导致 LLM 引用上下文之外的法条

改后 Rule 12：
  "如果参考资料中同时包含了某法条及其适用前提，必须一并说明。
   如果参考资料只包含法条本身却没有其适用前提，只需注明'具体适用条件
   请参考完整法规'，不要从自己的知识中补充前提条件。"
```

#### 3.2 加强"无法回答"的引导
```
在 ANSWER_STYLE_RULES 中增加：
  "当参考资料不足以完整、准确回答问题时，直接回答'根据现有资料无法确定'，
   这比给出不完整的答案更好。不要试图用自己的知识填补空白。"
```

---

### 第四阶段：V4 架构修复（预计提升 3-5 分）

#### 4.1 计算器结果纳入评估上下文
**文件：** `backend/app/core/generator/agent_graph.py`

当前 `calc_result` 包含法条引用但未加入 `context_docs`，RAGAS 评估时看不到这些引用。

修改方案：将 Calculator Agent 输出的 calc_result 作为 extra_context 追加到最终回答的 sources 列表中，确保评估时能匹配。

#### 4.2 收紧 Validator 的容错空间
**文件：** `backend/app/core/generator/agent_graph.py`

```python
# 删除或弱化这句话：
"参考资料仅为节选片段，不要求回答引用的每条法条都出现在节选中。"

# 替换为：
"回答中引用的每条法条必须有对应的参考资料支持。如果引用了参考资料中不存在的法条，
标记为不忠实，需要移除该引用后重新生成。"
```

---

## 预期效果

| 阶段 | 措施 | 预计 faithfulness 提升 |
|------|------|------------------------|
| 第一阶段 | 护栏修复 + 压缩器阈值 + 截断长度 | +10~15 分 → 0.70~0.76 |
| 第二阶段 | 检索增强 (TOP_K/RERANK/权重) | +10~15 分 → 0.80~0.86 |
| 第三阶段 | Prompt 优化 | +5~8 分 → 0.85~0.90 |
| 第四阶段 | V4 架构修复 | +3~5 分 → 0.88~0.93 |

**目标：从 0.61 提升到 0.85+**

---

## 验证方法

每完成一个阶段，运行评估验证：

```bash
cd backend
python run_v3_comparison.py --only agent --sample 10
```

关注指标：
- `faithfulness` — 目标 ≥ 0.85
- `context_recall` — 检索召回，目标 ≥ 0.75
- `hallucination_rate` — 目标 ≤ 0.15
