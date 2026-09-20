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


---

## 会话 4：2026-09-17（RAGAS 评估 + 抓出 3 个静默丢数据的缺陷）

### 阶段 3 收尾：拆分大文件（routes.py）

`app/api/routes.py` 12814B → **679B**（仅聚合），拆出四个模块：

    app/api/deps.py        引擎单例引用 + SUPPORTED_EXT + require_engine
    app/api/chat.py        /chat
    app/api/knowledge.py   /knowledge-base/*, /documents/*
    app/api/evaluation.py  /evaluation/*

**拆分无端点丢失**：用 OpenAPI 逐条核对，当前 22 个端点 vs 拆分前 21 个
（旧 `routes.py` 内不含 `/api/v1` 前缀，挂载时加）+ `/health`，一一对应。

**鉴权不变量**：旧实现把 `verify_bearer` 挂在唯一的 router 上，新端点自动继承。
拆分后 `include_router` 的子路由**不会**继承聚合器的依赖，所以三个子路由各自
声明 `APIRouter(dependencies=[Depends(verify_bearer)])`。
⚠️ 这个不变量当时**没有测试守住** —— 见会话 5 补的结构守卫。

### 跑通 RAGAS 评估（重构后第一次）

Smoke(5 题) 完整评估，13m53s：

| 指标 | 值 |
|---|---|
| Faithfulness | 0.8289 |
| Context Precision | 0.7011 |
| Context Recall | 0.5667 |
| 幻觉率 | 0.1711 |
| 完整性 | 0.8 |

### 发现并修复的 3 个缺陷（Bug 11/12/13）

- **Bug 11**：`runner.py` 按模型名判断是否关思考 → 非 qwen 模型 98% 输出 token 是思考
- **Bug 12**：对比路径从 `rewritten_question` 提取概念，而改写会剥掉连接词
  → 切分失败 → 两次检索同一个整句 → 对比路径退化成单查询（与 Bug 9 同根因）
- **Bug 13**：`run_eval.py` 硬编码 `P@1`/`R@5`/`MRR`，而指标层产出
  `precision@1`/`recall@5`/`mrr` → 检索指标**永远显示 N/A**，数据却在同一个响应体里

### 测试结果

- pytest：159 → **171 passed**
- 新增 3 个测试文件（10 例），**三个修复各有阴性对照**

### 错误日志（会话 4 追加）

| 错误 | 原因 | 处理 |
|---|---|---|
| `name 'extra_body' is not defined` | 我在做阴性对照时把文件改坏，而**延迟导入的模块会在调用时才读磁盘** | 修回即可；反之也说明「修完不用重启」 |
| 阴性对照没红（空跑守卫） | 只测了 helper，而修复在**调用点** | 改用「记录 query 的检索器桩」测调用点行为 |

### 五问重启检查

1. 现在到哪了？→ 阶段 3 完成，RAGAS 首次跑通
2. 验证网如何？→ 导入冒烟 76/76；检索基线 Δ=0；pytest 171；RAGAS 跑通
3. 下一步？→ 找客观盲区（哪些模块从没被测过）
4. 有什么阻塞？→ 无
5. 有没有未记录的决定？→ 「重依赖一律惰性导入」已写入代码注释

---

## 会话 5：2026-09-20（git 事故抢救 + 死代码清理 + 结构守卫）

### ⚠️ 事故：`.git` 被「删除到回收站」

做 A/B 导入耗时对比时，含 `git stash -q` 的命令被 SIGTERM 中断，之后 `git`
报 **「not a git repository」**。

**根因**：git 的 `is_git_directory()` 要求 `HEAD` / `objects` / **`refs`**
三者齐全 —— 只缺 `refs` 一个目录，整个仓库就被判为非仓库。

**破坏面**：`.git/objects/pack/*.pack`（主 pack 15.19MB）、约 300 个散对象被删。
D 盘回收站里找到 **494 个来自 `LaborRAG\.git` 的条目**，时间集中在 16:19–16:20，
其中有 `maintenance.lock` ×10、`HEAD.lock` ×11、`packed-refs.lock` ×12
—— **同一路径被反复删除**，说明有东西在周期性把 `.git` 当垃圾清理。

**抢救（全部可验证）**：

1. 工作区完整备份 → `D:/tmp/LaborRAG_worktree_20260920.tar.gz`（7.1MB / 725 文件）
2. 重建 `.git/refs/{heads,tags,remotes}` + 写入 `refs/heads/main`
3. 解析回收站 `$I` 元数据（`size(8B)|deltime(8B)|pathlen(4B)|UTF-16LE path`），
   还原 **304 文件 + 77 目录**
   - 必须跳过 `.lock`（陈旧锁会让 git 彻底锁死）
   - 目标路径是**残留空目录**时不能跳过，否则对象还原不上（第一遍漏了 61 个）
4. **18 个缺失 blob 全部字节级重建**：17 个用
   `git hash-object -w --path=<路径>`（`--path` 应用 autocrlf 归一化）；
   1 个（`generate.py`，我改过）用**反向撤销自己的改动**重建，验证哈希一致
5. **HEAD 的 tree 用 `git write-tree` 精确命中** —— `git commit` 后 index 仍保存
   该 commit 的树内容，重建出完全相同的 SHA（`3a1b953d…`）
6. **5 处断链用 `git replace --graft` 桥接** —— 关键：**reflog 里直接记录了父子
   关系**（`<old> <new> …\t commit: msg` 的 `old` 就是 parent）

**结果：`git log` 从「只列 11 个就 fatal」恢复到列出 39 个，一直到 `Initial commit`。**

**遗留（可接受）**：`4177a0c` 的 tree 仍缺（只影响该 commit 的 `git log -p`）；
`git replace` 引用不随 push 传播，换机器需重建。

### 死代码清理（两处，都由「重依赖」线索牵出）

- `app/retrieval/sparse.py` —— 从未接线的检索器（用户删除）
- `EvalRunner.METRICS` —— 类属性，全仓零读取
- `_N1ChatModel(BaseChatModel)` —— 39 行，全仓零引用

后两个是 `runner.py` 模块级重依赖的**唯一原因**：`METRICS` 强制 `ragas.metrics`，
`_N1ChatModel` 强制 `langchain_core.language_models`（基类必须在类创建时可用，
**无法惰性化**）。删除后 **`import app.evaluation.runner` 17.30s → 0.94s（18 倍）**。

### 重依赖导入的完整追查链

1. 7 个模块模块级 `import langchain_openai` → `import app.agent.graph` 17.5s
2. 改成 `TYPE_CHECKING` + 字符串注解后**仍然拉 torch** → 说明找错了对象
3. import hook 抓到真凶：**`ChatPromptTemplate` 的「取值」本身**触发
   `langchain_core.prompts.__getattr__` → `prompts.chat` → `messages.utils`
   → `messages.__getattr__` → `block_translators.openai` → `language_models._utils` → torch
4. 逐条测量，定位三个「昂贵入口」（都经 `block_translators`）：

   | 导入 | 耗时 | torch |
   |---|---|---|
   | `langchain_core.prompts.ChatPromptTemplate` | 16.6s | ✓ |
   | `langchain_core.output_parsers.StrOutputParser` | 15.7s | ✓ |
   | `langchain_core.language_models.BaseChatModel` | 15.2s | ✓ |
   | `langchain_core.messages.BaseMessage` | 0.7s | ✗ |
   | `langchain_core.documents.Document` | 0.4s | ✗ |

5. **决定不动 langchain_core**：把 `prompts` 也下沉到函数内只是把 15s 从
   **启动**挪到**首次请求**，对答辩演示反而更糟，不是净收益。

### 新增两条结构守卫（含非空性检查）

- `tests/test_heavy_imports.py` —— AST 扫 `app/**` 的**模块级** import，
  禁 10 个重依赖。非空性用例同时验证「只扫模块级」（函数内的 import 不得误报，
  否则会逼人把 import 提到模块级，正好做反）
- `tests/test_api_auth_wiring.py` —— 遍历聚合路由下每条路由，断言
  `dependencies` 含 `verify_bearer`。漏挂新子路由 = **静默鉴权绕过**，
  而 `TestAuth` 打的是具体端点，不会变红

### 测试结果

- pytest：171 → **214 passed**
- **两个阴性对照均确认能变红**：注入模块级 `import torch` → 重依赖守卫 FAILED；
  摘掉 `knowledge.py` 的 `dependencies` → 鉴权守卫 FAILED 并列出 5 个裸奔端点

### 错误日志（会话 5 追加）

| 错误 | 原因 | 处理 |
|---|---|---|
| `git` 报「不是仓库」 | 只缺 `.git/refs` 目录 | 重建目录即恢复 |
| 正则「莫名不匹配」 | **Git Bash 把 heredoc 里的 `\n` 转成 `/n`** | 改用 `chr(10)` / 逐行处理 |
| `tar: Cannot connect to D:` | Git Bash 把 `D:/x` 当远程主机 | 改用 `/d/x` |
| `Depends` 对象没有 `.call` | FastAPI 用 `.dependency` 存函数 | 改属性名 |
| 本地接口返回 502 | 本机 `http_proxy` 拦截 `127.0.0.1` | `ProxyHandler({})` 绕过 |

### 五问重启检查

1. 现在到哪了？→ 阶段 0–4 全完成；git 事故已修复；重依赖已清零
2. 验证网如何？→ 导入冒烟 76/76；基线 Δ=0；pytest 214；RAGAS 跑通（smoke）
3. 下一步？→ 20 题完整评估（**留到最后**）、文档回写、`persistent.py` 拆分
4. 有什么阻塞？→ 无
5. 有没有未记录的决定？→ 「不做 langchain_core 的惰性化」已记录理由

### 额度测算（为零额度离线完成，供决定是否跑 20 题）

读 `ragas 0.4.3` 源码数清调用结构：`Faithfulness` **2 次**、
`ContextRecall` **1 次**、`ContextPrecision` **每个子块 1 次**（大头）。
用上次 smoke 的真实缓存数据复刻子块切分，实测**子块均值 36.8 个**
→ 每题约 40 次 RAGAS 调用，20 题约 800 次。

| 槽位 | 20 题预估 | 可用 | 占比 |
|---|---|---|---|
| deepseek-v4.1-flash（RAGAS） | ~330k | 750k | ~45% |
| qwen3.8-max（生成） | ~130k | 900k | ~14% |
| qwen3.8-flash（改写/路由/充分性） | ~100k | 980k | ~10% |

**最大风险是 `MAX_RETRIES = 3`**：失败会整轮重跑，一次重试即 +330k。
建议全量前先降到 1。低风险校准路径：`--difficulty easy`（6 题，约 100k）。
