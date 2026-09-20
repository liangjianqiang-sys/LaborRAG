# 进度日志

## 会话 1：2026-09-16

**起因**：用户提出"检索层缺少测试，修改一个小部分以后不能反馈修改效果"，
随后转向"参考 agent-service-toolkit 重组本项目目录结构"。

### 阶段 0：建立验证网（行为基线）
- **状态：** ✅ complete（于会话 2 完成，见下）
- **开始时间：** 2026-09-16 20:17
- 执行的操作：
  - 通读项目：`README.md`、`config.py`、`main.py`、`rag_engine.py`、`agent_graph.py`、全部 8 个 node、
    `vectorstore.py`、`text_splitter.py`、`retriever/*`、`law_graph.py`、`evaluation/*`、`routes.py`、前端结构
  - 核对 README 宣称指标 → 在 `eval_unknown_20260602_173829.json` 找到完全对应的原始数据，**数字属实**
  - 侦察运行环境：发现 base 环境 torch 损坏，`RAG` conda 环境可用且依赖齐全
  - 确认模型已缓存、向量库已构建 → **离线检索 harness 可行**
  - 拉取 agent-service-toolkit 真实目录 tree（GitHub API）
  - enumerate 路径陷阱（`grep __file__|parents\[|sys\.path`）
  - 统计项目内部 import 关系（`app.config` 被引 19 次、`app.core.retriever.base` 10 次等）
- 创建/修改的文件：
  - `task_plan.md`（新建）
  - `findings.md`（新建）
  - `progress.md`（本文件，新建）
- **尚未产出**：`scripts/smoke_import.py`、`scripts/retrieval_baseline.py`、`baselines/pre_refactor.json`

### 阶段 1：目录重构（纯移动）
- **状态：** ⏸ **建议缓做**（会话 2 结论：风险最高、直接价值最低；且 40+ 处 import 改动
  与「不为对齐目录而重构」的既有建议相悖）
- 执行的操作：—
- 创建/修改的文件：—

### 阶段 2：消除重复实现
- **状态：** ✅ complete（于会话 2 完成，见下）
- 5 项中完成 4 项；「法律名 ↔ 文件名映射统一」经复核判定**不应合并**（两者方向相反，见会话 2）
- 「1–999 往返属性测试」经实测判定**不应做**（对零缺陷是假阴性）

### 阶段 3：拆分大文件
- **状态：** ✅ complete（于会话 2 完成，见下）
- `law_graph.py` 已拆为包；`routes.py` 拆分未做（价值低于前者）

### 阶段 4：收尾
- **状态：** ⏳ pending

---

## 会话 2：2026-09-16（续，四轮）

**起因**：用户连续下达「落地」→「加进 .gitignore 收尾」→「读取三份规划文件并复核」
→「完成吧」→「继续」×3。

### 阶段 0：验证网（完成）

- `scripts/smoke_import.py`：遍历 `app/**/*.py` 逐个 import，报告失败与慢导入（≥1s）。
  **56/56 通过**。顺带定位到 `app.core.document_loader` 模块级导入
  `langchain_community.document_loaders`（36.5s），修复后 `app.api.routes`
  从 **143.55s → 8.96s**。
- `scripts/retrieval_baseline.py`：离线确定性检索基线。
  **关键修正**：初版只调 `retriever.retrieve()`，漏掉整个知识图谱层 —— 而
  `concept_lookup` 是 P@1 的主要来源（实测「没签合同被辞了」精确命中
  `劳动合同法第82条`）。用初版做回归，重构把 `law_graph` 改坏了也不会报警。
  改为调用**真实生产节点** `retrieve_docs` + `classify_complexity`，传 `_NoLLM`
  桩让 HyDE/多查询扩展在既有 try/except 里降级为不增强。新增 `--compare` 模式。
- `baselines/pre_refactor.json` + `pre_refactor_reranked.json` 已落盘。

### 阶段 2：消除重复实现（完成 4/5）

| 计划项 | 状态 | 落地位置 |
|--------|------|---------|
| 法条编号正则统一 | ✅ | `app/core/law_refs.py`（非计划设想的 `app/utils/law_ref.py`） |
| 中文数字 ↔ 阿拉伯数字统一 | ✅ | 同上 |
| `file_lock` 下沉 | ✅ | `app/core/fs_utils.py` |
| 法律名 ↔ 文件名映射统一 | ❌ 判定不应合并 | — |
| 1–999 往返属性测试 | ❌ 判定不应做 | — |

### 阶段 3：拆分大文件（完成）

`law_graph.py`（757 行，62% 是纯数据表）拆为包 + 门面，**调用点零改动**。
`routes.py` 拆分未做。

### 发现并修复的 6 个真实 bug

1. **法条切分正则漏「零」**（README 已有）—— 修复不完整，见 #3
2. **检索指标空输入返回值不一致**（README 已有）
3. **同一字符类散落 7 处，只修了 1 处** —— `classify.py` 漏「零」、`retrieve.py` 漏「千」
4. **知识图谱注入顺序随进程变化 → 检索结果不可复现**（`law_graph` 直接迭代 set）
5. **响应字段取错 dict 键** —— `rewritten_question` 一直返回意图值
6. **中文数字解析散落 6 处** —— `第一百〇一条` 被算成第 100 条

### 测试结果

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| `pytest`（重构前基线） | `cd backend && pytest -q` | 全绿 | **121 passed**（会话 1 时仅 15） | ✅ |
| `smoke_import` | 遍历 `app/**` 全部模块 | 全部 import 成功 | **56/56 通过** | ✅ |
| `retrieval_baseline`（hybrid） | 20 题 Golden Set | P@1 ≈ 0.9 | **P@1 0.8500 / MRR 0.9250 / MAP 0.7384** | ✅ |
| `retrieval_baseline`（reranked 消融） | 同上 | 数字应变化（敏感性验证） | R@5 0.7750→0.8500、MAP 0.7384→0.7778 | ✅ |
| 基线可复现性 | 同配置连跑两遍 | Δ 全为 0 | **六项 Δ 全 0.0000，rank-1 逐题一致** | ✅ |

> 阶段 0 的三个"重构前"结果已在任何文件移动之前拿到；`baselines/pre_refactor.json`
> 在阶段 2/3 的每次改动后都用 `--compare` 复核，**Δ 始终为 0**。

### 错误日志（会话 2 追加）

| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-09-16 | `Path.unlink()` 被沙箱「安全删除」垫片拦截（`SHFileOperationW 失败: 0x2`） | 1 | 文件实际已移走；`__pycache__` 需单独 `rm -rf` 清理 |
| 2026-09-16 | 断言「lookup 模块不该有数据表」用 `hasattr` 判断失败 | 1 | `hasattr` 对 import 进来的名字也为真；改用 AST 判断是否在本模块**定义** |
| 2026-09-16 | 基线连跑两遍 MAP 在 0.7294/0.7301 抖动 | 2 | 固定 `PYTHONHASHSEED=0` 定位到 set 迭代序 → 见 bug #4 |

### 五问重启检查

| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 0/2/3 已完成并通过验证；阶段 1 建议缓做，阶段 4 未开始 |
| 我要去哪里？ | 阶段 4 收尾（tests/ 镜像结构、README 结构章节、`__pycache__` 残留、`sparse.py` 标注 experimental）；或按用户意愿转向 |
| 目标是什么？ | 把 `backend/app` 按能力/职责重新分层，消除重复实现与分层倒置，全程零行为变化 |
| 我学到了什么？ | 「验证网必须测生产依赖的那一层」；「同一概念的多份实现会静默漂移」；「同进程测不出哈希序依赖」；「往返属性测试对归一化缺陷是假阴性」 |
| 我做了什么？ | 建验证网 + 修 6 个 bug + 4 处架构收敛（正则/中文数字/检索器门面/分层倒置）+ law_graph 拆包；测试 15 → 121 |

---

## 会话 3：2026-09-16（续，阶段 1 目录重构）

**起因**：用户在明确询问「是否执行阶段 1」后选择了执行。前五轮我一直建议缓做
（风险最高、直接价值最低），但风险那一半确实已被验证网兜住。

### 阶段 1：目录重构（完成）

49 个文件 `git mv`，225 处 import 替换（46 文件），`app/core` 这个「6 种职责混在一起的
杂物抽屉」被拆开。新结构见 `task_plan.md` 的文件映射表与 README 的项目结构章节。

**4 个路径陷阱全部按计划预判命中**，处理方式：

| # | 陷阱 | 处理 |
|---|------|------|
| 1 | `config.py` 的 `BASE_DIR` 深度 +1 | 改用 `_THIS_FILE.parents[3]` 写法，并显式导出 `BACKEND_DIR`（`parents[2]`） |
| 2 | `bm25.py` 的 `law_dict.txt` 路径 | 改为基于 `BACKEND_DIR`；抽出 `_law_dict_path()` 并**缺失即抛 FileNotFoundError**（原来是 `if os.path.exists` 静默跳过） |
| 3 | `run.py` 的 `logging_config` 导入 | 批量替换已覆盖 |
| 4 | `evaluation` 三个用 `__file__` 推算根目录的文件 | 计划映射本身已保持同深度，无需改 |

### 搬家引入的两个包级循环依赖（计划未预判，复核时发现并修掉）

- **`tools ↔ agent`**：`labor_calculator` 从 `agent.constants` 取计算常量，而
  `agent/nodes/calculate` 又用 `labor_calculator`。→ 两个常量只服务于计算，移进
  `labor_calculator.py` 本身。
- **`utils ↔ retrieval`**：`utils/text.py::_law_display()` 需要 `_SOURCE_LAW_MAP`，而
  `retrieval/domain_boost` 需要 `utils/law_refs`。→ 数据表下沉到 `utils/law_refs.py`
  （最底层），`domain_boost/source_map.py` 改为再导出。

修完后分层**零反向依赖**，由 `test_layering.py`（重写，5 → 17 例）持续守卫。

### 测试结果

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| `pytest` | `cd backend && pytest -q` | 全绿 | **136 passed**（会话 2 时 121） | ✅ |
| `smoke_import` | 遍历 `app/` + `scripts/` | 全部 import 成功 | **66/66 通过**（会话 2 时 63，已扩展到覆盖 scripts） | ✅ |
| `retrieval_baseline --compare` | 20 题 Golden Set | Δ 全为 0 | **六项 Δ 全 0.0000，rank-1 逐题一致** | ✅ |
| 真实服务启动 | `python run.py` + `/health` | `knowledge_base_ready: true` | ✅ 另验证 `/api/v1/knowledge-base/status` → `total_chunks: 1243` | ✅ |

> 陷阱 1/2 只会在运行时暴露（`BASE_DIR` 算错不会报错、词典静默不加载不会报错），
> 所以「真起一次服务」是这一阶段的必要验证步骤，不能只看单测。

### 错误日志（会话 3 追加）

| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-09-16 | 49 次 `git mv` + 2 次 `git rm` = 51 次删除类操作，超过沙箱单轮 50 次阈值 → 索引里的改名成功但**工作树文件被隔离**，`backend/app/` 整个消失 | 1 | `git checkout-index -a -f` 从索引恢复；教训：大批量文件操作要分批并核对文件数 |
| 2026-09-16 | 测试里 `from app.retrieval import law_graph as lg` —— `law_graph` 是**被导入的名字**而非模块路径，前缀替换覆盖不到 | 1 | 由测试抓出（6 例变红）；单独修正 3 处 |
| 2026-09-16 | `test_calculators.py` 从 `app.agent.constants` 取计算常量，常量移走后收集失败 | 1 | 改为从 `app.tools.labor_calculator` 导入 |
| 2026-09-16 | 用 `grep -E "FAILED\|passed\|failed"` 过滤 pytest 输出，**漏掉了 "error"**，导致收集错误被静默吞掉两次 | 2 | 过滤模式加入 `error\|ERROR`；改用 `tail` 看完整汇总 |

### 五问重启检查

| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 0/1/2/3/4 全部完成，工作区干净 |
| 我要去哪里？ | 计划已执行完毕。后续可选：`routes.py` 拆分、前端运行时验证、`sparse.py` 接线 |
| 目标是什么？ | 把 `backend/app` 按能力/职责重新分层，消除重复实现与分层倒置，全程零行为变化 |
| 我学到了什么？ | 「包级循环依赖往往是搬家引入的，因为搬的人只看文件归属不看依赖方向」；「空跑的架构测试比没有更糟」；「沙箱有批量删除阈值，超了会静默隔离文件」 |
| 我做了什么？ | 完成阶段 1；累计 7 个 commit、7 个 bug、5 处架构收敛、2 处循环依赖；测试 15 → 136 |

---

*每个阶段完成后或遇到错误时更新此文件*
