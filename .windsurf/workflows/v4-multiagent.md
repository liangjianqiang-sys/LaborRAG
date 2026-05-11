---
description: V4多Agent版开发 - 多Agent协作架构
---

# V4 多Agent版开发工作流

## 前置条件
- V3高级版已完成且可正常运行

## 当前状态
- [ ] 未开始（需先完成V3）

## 架构设计

```
用户提问
   │
   ▼
┌──────────┐
│ Router   │ ← 路由Agent：判断问题类型
│  Agent   │   (法条查询/案例分析/计算类/对比类)
└────┬─────┘
     │ 根据类型分派
     ├──────────────┬──────────────┐
     ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│Retriever│  │Calculator│  │Comparator│
│ Agent   │  │  Agent   │  │  Agent   │
│检索+纠错 │  │赔偿/补偿  │  │法律对比   │
│(V3 CRAG)│  │金额计算   │  │新旧法对比 │
└────┬────┘  └────┬─────┘  └────┬─────┘
     │            │              │
     └────────────┴──────────────┘
                  │
                  ▼
           ┌──────────┐
           │ Validator │ ← 验证Agent：检查回答合法性
           │  Agent    │   (是否忠实、是否有法律依据)
           └────┬─────┘
                │
                ▼
              最终回答
```

## 开发步骤

### 阶段1: Agent基类与路由设计
1. 安装新依赖: langgraph (如V3未安装)
2. 创建 `backend/app/core/agents/` 目录
3. 创建 `base.py` — Agent抽象基类
4. 创建 `router.py` — Router Agent (意图识别+路由)

### 阶段2: Retriever Agent
5. 创建 `retriever.py` — 封装V3 CRAG为Agent
6. 支持检索+评估+纠错闭环

### 阶段3: Calculator Agent
7. 创建 `calculator.py` — 法律金额计算Agent
8. 实现计算工具: 经济补偿金/加班费/工伤赔偿/年休假工资
9. LLM调用工具进行精确计算

### 阶段4: Validator Agent
10. 创建 `validator.py` — 回答合法性验证Agent
11. 检查回答是否有法律依据
12. 检查是否存在法律幻觉

### 阶段5: 多Agent编排
13. 创建 `orchestrator.py` — LangGraph多Agent编排
14. 定义Agent间通信协议
15. 实现Agent协作流程

### 阶段6: 前端更新
16. Agent执行过程可视化
17. 显示当前执行Agent和状态
18. 工作流模式切换(V1/V2/V3/V4)

### 阶段7: 消融实验
19. V1 vs V2 vs V3 vs V4四版对比
20. 多Agent各组件消融实验
21. Router准确率评估

## 验收标准
- Router能正确分类问题类型(准确率>85%)
- Calculator能精确计算法律金额(误差<1%)
- Validator能有效检测法律幻觉
- 四版对比实验数据清晰
