# -*- coding: utf-8 -*-
"""第三轮多统计量框架（1230 起）：只替换 STAT 一行，SQL 逐字节不变（评测机按 SQL 文本缓存）。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1230_同时段季节性二十日.py")
t0 = io.open(BASE, encoding="utf-8").read()
VAR = [
    (1231, "L_lab40", "同时段季节性四十日"), (1232, "L_lab10", "同时段季节性十日"),
    (1233, "L_ret20", "同时段收益季节性二十日"), (1234, "L_lab20_dm", "同时段季节性二十日去时段均值"),
    (1235, "M_q", "聪明钱VWAP偏离"), (1236, "M_q_lin", "聪明钱线性权重"),
    (1237, "N_dir", "大单方向"), (1238, "N_big", "大单减小单分钟收益"),
    (1239, "Q_eff", "路径效率"), (1240, "Q_seff", "带符号路径效率"), (1241, "Q_ac1", "分钟收益自相关"),
    (1242, "U_nj", "跳跃次数"), (1243, "U_sj", "带符号跳跃收益"),
    (1244, "S_osz", "买卖每笔挂单量比"), (1245, "S_osz30", "买卖每笔挂单量比近三十分钟"),
    (1246, "O_q5", "一档队列比近五分钟"), (1247, "T_acc", "放量加速乘方向"),
    (1248, "R_round", "整数元关口距离"), (1249, "R_half", "半元关口距离"), (1250, "V_spr", "当日相对价差"),
    (1251, "P_ac20", "个股半小时收益自相关二十日"), (1252, "P_ac60", "个股半小时收益自相关六十日"),
]
for num, stat, name in VAR:
    a = 'STAT = "L_lab20"'
    assert t0.count(a) == 1
    t = t0.replace(a, 'STAT = "%s"' % stat, 1)
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write(t)
    print("wrote", num, stat, name)
