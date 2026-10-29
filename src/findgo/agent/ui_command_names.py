"""命令名清单。

单独放一个模块，让 Agent 层判断「这是不是命令」时不必导入 PyQt5。
"""

COMMAND_NAMES = frozenset(
    {
        "/diary",
        "/xing",
        "/compact",
        "/loop",
        "/hana-plugin-creator",
        "/office-documents",
    }
)
