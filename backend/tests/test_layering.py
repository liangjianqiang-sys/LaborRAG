"""分层依赖方向测试。

目录重构后 `app/` 的分层（低 → 高）：

    core / schemas / utils                    最底层：配置、DTO、零依赖原语
    knowledge / retrieval / memory / tools    领域层：知识库、检索、记忆、工具
    agent                                     Agentic RAG 工作流
    services                                  应用服务（RAG 引擎编排）
    evaluation                                评估体系
    api                                       HTTP 层
    main                                      入口

规则：**低层不得 import 高层**。

这类问题不报错、不影响功能，只会让「改 A 层却炸 B 层」变成常态，所以用静态
扫描把它变成会变红的规则。扫描器走 AST，**函数内的延迟导入同样能扫到** ——
反向依赖往往正是藏在那种「为了绕开循环依赖」的延迟导入里。本项目真实发生过
两次：`core → evaluation`（取 `file_lock`）、`utils → retrieval`（取 `_SOURCE_LAW_MAP`），
都是延迟导入，都是不报错的静默耦合。
"""
import ast
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"

# 从低到高。索引越大层级越高。
LAYERS = [
    "core",
    "schemas",
    "utils",
    "knowledge",
    "retrieval",
    "memory",
    "tools",
    "agent",
    "services",
    "evaluation",
    "api",
]
_RANK = {name: i for i, name in enumerate(LAYERS)}


def _layer_of(path: Path) -> str:
    """文件所属的顶层包名；app/main.py 这类根文件返回 '(root)'。

    对 tmp_path 里的测试样本（不在 APP_DIR 下）回退为「直接父目录名」，
    这样防桩失效用例才能复用同一套判定逻辑。
    """
    try:
        parts = path.relative_to(APP_DIR).parts
    except ValueError:
        return path.parent.name
    return parts[0] if len(parts) > 1 else "(root)"


def _iter_imports(path: Path):
    """产出文件里所有 import 的模块名与行号（含函数体内的延迟导入）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # level != 0 是相对导入，按本包内处理，不参与跨层判断
            if node.module and node.level == 0:
                yield node.module, node.lineno


def _rel(path: Path) -> str:
    """相对 backend/ 的路径；测试样本（tmp_path）回退为文件名。"""
    try:
        return path.relative_to(BACKEND_DIR).as_posix()
    except ValueError:
        return path.name


def _scan_dir(directory: Path) -> list[str]:
    """扫描目录下所有文件，返回「低层依赖高层」的违规列表。"""
    violations = []
    for path in sorted(directory.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        src_layer = _layer_of(path)
        src_rank = _RANK.get(src_layer)
        if src_rank is None:
            continue  # 根文件（main.py）作为入口，允许依赖任意层
        for module, lineno in _iter_imports(path):
            if not module.startswith("app."):
                continue
            tgt_layer = module.split(".")[1]
            tgt_rank = _RANK.get(tgt_layer)
            if tgt_rank is not None and tgt_rank > src_rank:
                violations.append(
                    f"{_rel(path)}:{lineno} （{src_layer} → {tgt_layer}）{module}"
                )
    return violations


@pytest.mark.parametrize("layer", LAYERS)
def test_不允许反向依赖上层(layer):
    violations = _scan_dir(APP_DIR / layer)
    assert not violations, (
        f"app/{layer} 出现反向依赖上层的 import：\n  " + "\n  ".join(violations)
    )


def test_只有入口可以依赖_api_层():
    """除 app/main.py（入口）外，任何模块都不应 import app.api。

    api 是最上层；被下层引用会立刻形成环（api → services → ... → api）。
    """
    offenders = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if _layer_of(path) in ("api", "(root)"):
            continue  # api 层自身与入口
        for module, lineno in _iter_imports(path):
            if module == "app.api" or module.startswith("app.api."):
                offenders.append(
                    f"{path.relative_to(BACKEND_DIR).as_posix()}:{lineno} → {module}"
                )
    assert not offenders, "app.api 被非 api 层引用了：\n  " + "\n  ".join(offenders)


def test_底层原语模块保持零依赖():
    """app/utils/law_refs.py 与 app/utils/files.py 必须零 app 依赖。

    它们位于依赖图最底端，被 knowledge / retrieval / agent 等各层引用。
    一旦引入 app.* 依赖，就会立刻把整张图拉成环。
    """
    for name in ("law_refs.py", "files.py"):
        path = APP_DIR / "utils" / name
        deps = [m for m, _ in _iter_imports(path) if m.startswith("app")]
        assert not deps, f"app/utils/{name} 出现了 app 依赖：{deps}"


def test_延迟导入的违规同样能被扫到(tmp_path):
    """反向依赖常藏在函数内的延迟导入里（为了绕开循环依赖），必须也能扫到。"""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n"
        "    from app.evaluation.utils import sanitize_floats\n"
        "    return sanitize_floats({})\n",
        encoding="utf-8",
    )
    # 用一个「core 层」目录包住它，让 _layer_of 能判定层级
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "probe.py").write_text(probe.read_text(encoding="utf-8"), encoding="utf-8")
    violations = _scan_dir(core_dir)
    assert violations, "扫描器漏掉了函数体内的延迟导入"
    assert "app.evaluation" in violations[0]


def test_扫描器本身有效_能发现违规(tmp_path):
    """防桩失效：构造一个必然违规的假层，确认扫描逻辑真的会报。"""
    fake = tmp_path / "core"
    fake.mkdir()
    (fake / "bad.py").write_text("import app.evaluation.runner\n", encoding="utf-8")
    assert _scan_dir(fake), "扫描器没能发现明显的反向依赖"


def test_扫描器不会误报合法依赖(tmp_path):
    """反向验证：低层引用更低层 / 同层 / 共享叶子都不该被算作违规。"""
    fake = tmp_path / "retrieval"
    fake.mkdir()
    (fake / "ok.py").write_text(
        "import app.core.config\n"
        "from app.schemas.chat import ChatRequest\n"
        "from app.utils.law_refs import cn_to_int\n"
        "from app.knowledge.store import VectorStoreManager\n",
        encoding="utf-8",
    )
    assert _scan_dir(fake) == []


def test_分层清单与磁盘上的包一致():
    """防呆：若新增/改名了顶层包却忘了登记进 LAYERS，这条会变红。"""
    actual = {
        p.name
        for p in APP_DIR.iterdir()
        if p.is_dir() and p.name != "__pycache__" and (p / "__init__.py").exists()
    }
    assert actual == set(LAYERS), (
        f"app/ 下的顶层包与 LAYERS 不一致：\n"
        f"  磁盘有但清单没有：{sorted(actual - set(LAYERS))}\n"
        f"  清单有但磁盘没有：{sorted(set(LAYERS) - actual)}"
    )
