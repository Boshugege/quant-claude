@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start F2_fine_aux15 >> e2e\runs\q4090.log
set GB_VIEW=fine& env\python.exe e2e\scripts\gbdt_member.py F2_fine_aux15 --trees 800 --depth 8 --lr 0.03 --rows 0.5 --cols 0.7 --min-leaf 1000 --fields full --norm rel3 --tgt e16 --aux-w 1.5 --win shape,book,tod > e2e\runs\F2_fine_aux15.out 2>&1
echo %date% %time% end F2_fine_aux15 >> e2e\runs\q4090.log
