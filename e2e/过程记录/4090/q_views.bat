@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start S1_slow >> e2e\runs\q4090.log
set GB_VIEW=slow& env\python.exe e2e\scripts\gbdt_member.py S1_slow --trees 800 --depth 8 --lr 0.03 --rows 0.5 --cols 0.7 --min-leaf 1000 --fields full --norm rel3 --tgt e16 --win shape,book,book2,book3,tod,pos,state,cls,mkt,lead,dow,auc > e2e\runs\S1_slow.out 2>&1
echo %date% %time% end S1_slow >> e2e\runs\q4090.log
echo %date% %time% start F1_fine >> e2e\runs\q4090.log
set GB_VIEW=fine& env\python.exe e2e\scripts\gbdt_member.py F1_fine --trees 800 --depth 8 --lr 0.03 --rows 0.5 --cols 0.7 --min-leaf 1000 --fields full --norm rel3 --tgt e16 --win shape,book,tod > e2e\runs\F1_fine.out 2>&1
echo %date% %time% end F1_fine >> e2e\runs\q4090.log
echo %date% %time% views done >> e2e\runs\q4090.log
