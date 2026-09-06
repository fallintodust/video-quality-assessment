# -*- coding: utf-8 -*-
"""wandb 离线桩模块：Windows 本地复现无需登录/联网，把 init/log 变成 no-op。

放在本目录（sys.path[0]）即可遮蔽真正的 wandb，training_with_divide.py
中的 `import wandb` 会优先命中本文件。
"""


class Settings:
    def __init__(self, *args, **kwargs):
        pass


def init(*args, **kwargs):
    return None


def log(*args, **kwargs):
    pass
