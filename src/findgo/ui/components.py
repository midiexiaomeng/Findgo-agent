"""基础组件：日志桥、日志 Handler、后台线程。"""

from __future__ import annotations

import logging

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class LogBridge(QObject):
    """把 logging 记录转成 Qt 信号，供界面线程消费。"""

    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:
            pass


class Worker(QThread):
    """在后台线程跑一个可调用对象，结果通过信号回传。"""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self) -> None:
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as e:
            self.failed.emit(str(e))
