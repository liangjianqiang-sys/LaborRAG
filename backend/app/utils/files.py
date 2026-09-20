"""跨模块的通用文件工具（零依赖，属于最底层）。

为什么单独放一个模块
--------------------
`file_lock` 是「串行化文件写入」的通用原语，最初定义在 `app/evaluation/utils.py`。
于是最底层的 `app/core/vectorstore.py` 不得不反向 import 评估层：

    app/core/vectorstore.py → app/evaluation/utils.py

依赖方向由此颠倒 —— core 无法脱离 evaluation 独立存在，动评估层的目录结构
就会波及核心层。放到 core 最底层后，方向恢复为 `core ← evaluation`。
"""
import threading

# 文件写入互斥锁：并发写 JSON / JSONL 时避免内容交错。
# 注意是**全局单例**，所有调用方共享同一把锁（不要在各模块各自 new 一把，
# 那等于没有互斥）。
file_lock = threading.Lock()
