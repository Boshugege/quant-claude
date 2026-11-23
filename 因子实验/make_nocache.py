# -*- coding: utf-8 -*-
"""为平台内存实测生成"避开评测机 SQL 缓存"的副本：在每段 f-string SQL 开头插入一行注释（SQL 语义不变，缓存键改变）。
用法：python3 因子实验/make_nocache.py <因子文件> <输出文件> [标签]"""
import io, re, sys
src, dst = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "nocache"
s = io.open(src, encoding="utf-8").read()
s, n = re.subn(r'(=\s*f?"""\n)(\s*)(WITH|SELECT)', lambda m: m.group(1) + m.group(2) + "-- " + tag + "\n" + m.group(2) + m.group(3), s)
io.open(dst, "w", encoding="utf-8").write(s)
print(dst, "SQL 段数", n)
