# -*- coding: utf-8 -*-
"""把 因子代码/xxx.py 生成同名的 .ipynb（只保留 main 所在的代码单元，去掉 __main__ 自测块）。

用法：python3 因子实验/make_ipynb.py 因子代码/113_xxx.py
"""
import json
import os
import sys

SRC = sys.argv[1]
src = open(SRC, encoding="utf-8").read()
# 只删掉 __main__ 自测块本身（到下一个顶层 def／class 为止），块后面的函数一律保留。
# 10-19 教训：以前从自测块处整段截断，追加在文件末尾的辅助函数被丢掉，1620／1622 平台报错、浪费两次提交。
import re
cut = src.find('\nif __name__ == "__main__":')
if cut >= 0:
    m = re.search(r"\n(def |class )", src[cut + 1:])
    end = cut + 1 + m.start() + 1 if m else len(src)
    src = (src[:cut + 1].rstrip() + "\n\n\n" + src[end:]).rstrip() + "\n"
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "id": "main",
               "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.splitext(SRC)[0] + ".ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("写出", out)
