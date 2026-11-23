#!/bin/bash
# E3（10-19）：1603 方向的平台口径重新定权，岭回归方向改用多期目标（w = 0.25／0.5），λ = 1／2／3；之后加 1603 的 CLS20 与 1603 平台口径比五窗口（c23）。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
LAB=L_lab10,M_q_lin,N_dir,W_vdir,W_zero,Z_mfr60,Z_mfr90,s155,s139,s161,k1381,k1385,k1406,r1391,r1406,r1407,r1409
for w in 0.25 0.50; do
  t=${w/./}
  PYTHONPATH=. PYTHONWARNINGS=ignore $PY g_opt.py --base-main 1015-203626-f5f0 --base-o21 1017-102020-9a7c --base-o20 1017-102056-e592 --labels $LAB --penalties 1,2,3 --sub Y_sd5:0.10,X_n5d:-0.10 --aux-w $w --out $L/g4aux$t.json > $L/g4aux$t.log 2>&1 &
done
wait
for w in 025 050; do echo "== aux $w" >> $L/c23aux.log; PYTHONPATH=. PYTHONWARNINGS=ignore $PY c23_g4r_vs1603.py $L/g4aux$w.json >> $L/c23aux.log 2>&1; done
echo done > $L/e3.done
