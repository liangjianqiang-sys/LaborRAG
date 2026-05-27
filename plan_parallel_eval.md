# 评估并行化方案

## 现状分析

当前所有LLM调用均为**串行**，主要瓶颈：

| 阶段 | 耗时占比 | 当前方式 | 瓶颈 |
|------|---------|---------|------|
| Phase 1: 生成答案 | 60-70% | 逐题串行 `rag_engine.chat()` | 每题等LLM返回才处理下一题 |
| Phase 2: 检索指标 | <1% | 纯计算，无LLM调用 | 无需优化 |
| Phase 3: RAGAS评分 | 25-30% | ragas内部已有并行 | 无需优化 |
| Phase 3: 完整性评估 | 5-10% | 逐题串行 `completeness_llm()` | 可并行但收益较小 |

**核心加速点**：Phase 1 的逐题答案生成，从串行改为并发。

---

## 方案设计

### 1. 通用并发工具

在 `evaluation/utils.py` 中添加并发执行器：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_CONCURRENT = 4  # 并发度，避免API限流

def parallel_map(fn, items, max_workers=MAX_CONCURRENT, desc=""):
    """并发执行 fn(item)，返回按原始顺序的结果列表。
    
    每个item独立try/except，单题失败不影响其他。
    """
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e  # 交给调用方处理
    return results
```

### 2. EvalRunner._execute 并行化

**文件**: `eval_runner.py`

Phase 1 改为并发生成答案：

```python
# 原代码（串行）：
for i, item in enumerate(dataset, 1):
    response = self.rag_engine.chat(ChatRequest(question=item["question"]))
    ...

# 改为（并发）：
def _gen_answer(self, item):
    """单题答案生成，异常返回None。"""
    try:
        return self.rag_engine.chat(ChatRequest(question=item["question"]))
    except Exception as e:
        print(f"  生成失败: {item['question'][:30]}... → {e}")
        return None

responses = parallel_map(self._gen_answer, dataset, desc="Generating answers")

for i, (item, response) in enumerate(zip(dataset, responses)):
    if response is None:
        continue  # 跳过失败题目
    questions.append(item["question"])
    answers.append(response.answer)
    ...
```

Phase 3 `completeness_llm` 也可并发，但收益较小（仅5-10%耗时），暂不改动。

### 3. eval_persistent._execute_task 并行化

**文件**: `eval_persistent.py`

逐题评估改为并发：

```python
# 原代码（串行）：
for idx in sorted(indices_to_eval):
    record = self._evaluate_single(idx, item, eval_runner)
    _append_result(task_id, record)
    self._update_task_progress(task_id, idx, record["status"] == "success")

# 改为（并发 + 按序保存）：
def _eval_one(self, idx_item_evalrunner):
    idx, item, eval_runner = idx_item_evalrunner
    return idx, self._evaluate_single(idx, item, eval_runner)

items_to_eval = [(idx, dataset[idx], eval_runner) for idx in sorted(indices_to_eval)]
records = parallel_map(self._eval_one, items_to_eval, desc="Evaluating")

for idx, record in records:
    if isinstance(record, Exception):
        continue  # 不应发生，_evaluate_single内部已catch
    _append_result(task_id, record)
    self._update_task_progress(task_id, idx, record["status"] == "success")
```

### 4. 并发度控制

| 场景 | 建议并发度 | 理由 |
|------|-----------|------|
| DeepSeek API | 4 | 避免触发限流（默认5 RPM） |
| 本地/高限额API | 8 | 充分利用I/O等待 |

通过环境变量 `EVAL_MAX_CONCURRENT` 可配置，默认4。

### 5. 线程安全分析

| 组件 | 线程安全 | 说明 |
|------|---------|------|
| `ChatOpenAI.invoke()` | ✅ | LangChain LLM实例线程安全 |
| `FAISS/Chroma检索` | ✅ | 只读操作，线程安全 |
| `_append_result()` | ⚠️ | 文件追加写需加锁 |
| `_update_task_progress()` | ⚠️ | JSON读写需加锁 |
| `_save_task()` | ⚠️ | 同上 |

**解决方案**：为文件操作添加线程锁：

```python
_file_lock = threading.Lock()

def _append_result(task_id, record):
    with _file_lock:
        ...  # 原有追加逻辑

def _update_task_progress(task_id, idx, success):
    with _file_lock:
        ...  # 原有更新逻辑
```

---

## 预期收益

| 场景 | 串行耗时 | 并行耗时(4并发) | 加速比 |
|------|---------|----------------|-------|
| 20题评估 | ~8min | ~2.5min | 3.2x |
| 10题评估 | ~4min | ~1.2min | 3.3x |
| 5题评估 | ~2min | ~0.7min | 2.9x |

---

## 实施步骤

1. `utils.py` — 添加 `parallel_map` + `MAX_CONCURRENT` + `_file_lock`
2. `eval_runner.py` — Phase 1 改用 `parallel_map` 并发生成答案
3. `eval_persistent.py` — 逐题评估改用 `parallel_map`，文件操作加锁
4. `config.py` — 添加 `EVAL_MAX_CONCURRENT` 配置项
5. 测试验证 — 运行5题评估，确认结果正确且速度提升

## 风险与注意

- **API限流**：并发度过高可能触发429，默认4保守安全
- **结果顺序**：`parallel_map` 保证结果按原始顺序返回
- **断点续评**：并发模式下失败题目仍然正确记录，续评逻辑不变
- **RAGAS评分**：ragas内部已有并行机制，无需改动
