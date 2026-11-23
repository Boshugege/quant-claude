@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
set GB_VIEW=fine& env\python.exe e2e\scripts\gbdt_member.py T_fine_small --small --trees 20 --depth 6 --fields full --norm rel3 --tgt e16 --win shape,book,tod > e2e\runs\T_fine_small.out 2>&1
set GB_VIEW=slow& env\python.exe e2e\scripts\gbdt_member.py T_slow_small --small --trees 20 --depth 6 --fields full --norm rel3 --tgt e16 --win shape,book,book2,book3,tod,pos,state,cls,mkt,lead,dow,auc > e2e\runs\T_slow_small.out 2>&1
