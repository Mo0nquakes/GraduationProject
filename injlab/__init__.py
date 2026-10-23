"""injlab —— 提示注入检测器的对抗鲁棒性评测实验台（本科毕设骨架）。

设计原则：
1. 核心零依赖：只用标准库，`python -m injlab demo` 开箱即跑。
2. 重型依赖（sklearn / fastapi / torch）全部延迟导入，缺了只降级不崩。
3. 数据库 schema 是唯一真源：ER 图由 schema 生成，界面是 schema 的视图。
"""
__version__ = "0.1.0"
