"""UI 组件包。

职责边界：
- 只做渲染与事件转发，不写检索、不写 Agent 调度
- 业务动作通过信号或回调交给 MainWindow / Orchestrator
"""
