---
description: V2增强版开发 - 混合检索+重排序+RAGAS评估
---

# V2 增强版开发工作流

## 前置条件
- V1基础版已完成且可正常运行

## 当前状态
- [ ] 未开始（需先完成V1）

## 开发步骤

### 阶段1: Hybrid Search 混合检索
1. 安装新依赖: jieba (中文分词), rank-bm25
2. 创建 `backend/app/core/retriever/bm25.py` - BM25关键词检索实现
3. 创建 `backend/app/core/retriever/hybrid.py` - 混合检索实现:
   - 向量检索 + BM25检索并行执行
   - Reciprocal Rank Fusion (RRF) 融合两路结果
   - 可配置两路权重比例
4. 在 `config.py` 新增混合检索配置项:
   - RETRIEVER_TYPE: "vector" | "hybrid" | "reranked"
   - BM25_TOP_K, VECTOR_TOP_K, RRF_K

### 阶段2: Reranker 重排序
5. 安装新依赖: sentence-transformers (用于Cross-Encoder)
6. 创建 `backend/app/core/retriever/reranked.py` - 带重排序的检索:
   - 先用Hybrid检索获取候选集（如Top20）
   - 用BGE-Reranker-v2-m3对候选集精排
   - 返回精排后的Top-K结果
7. 在 `config.py` 新增重排序配置项:
   - RERANKER_MODEL_NAME
   - RERANK_TOP_K

### 阶段3: 法条结构化切分
8. 创建 `backend/app/core/text_splitter.py` - 法条级切分器:
   - 识别法条结构（第X条、第X款）
   - 按法条段落切分，保持法条完整性
   - 保留法条编号作为元数据
9. 修改 `vectorstore.py` 支持选择切分策略:
   - "recursive": 原有的固定长度切分
   - "law_article": 法条结构化切分

### 阶段4: RAGAS 评估体系
10. 安装新依赖: ragas, datasets
11. 创建 `backend/app/evaluation/` 目录:
    - `__init__.py`
    - `eval_dataset.py` - 评估数据集构建（人工标注问答对）
    - `eval_runner.py` - RAGAS评估执行器
    - `eval_report.py` - 评估报告生成（对比图表）
12. 创建评估API:
    - POST /api/v1/evaluation/run - 运行评估
    - GET /api/v1/evaluation/report - 获取评估报告
13. 准备评估数据集: 20-30个劳动法问答对（人工标注标准答案）

### 阶段5: 对比实验
14. 运行三组对比实验:
    - 纯向量检索 vs 混合检索 vs 混合检索+重排序
    - 固定长度切分 vs 法条结构化切分
    - 不同Embedding模型对比（可选）
15. 生成评估报告和对比图表

### 阶段6: 前端更新
16. 前端新增评估结果展示页面
17. 设置面板新增检索策略切换（vector/hybrid/reranked）
18. 来源卡片显示相关度分数

## 验收标准
- [ ] Hybrid Search检索结果优于纯向量检索
- [ ] Reranker重排序后Top-K精确度提升
- [ ] 法条切分保持法条完整性，无断裂
- [ ] RAGAS评估能正常运行并输出指标
- [ ] 对比实验数据清晰展示V2优于V1
