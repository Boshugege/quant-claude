D:\quant-claude\env\python.exe -m ensurepip --upgrade > pip_install.log 2>&1  
D:\quant-claude\env\python.exe -m pip install --upgrade pip >> pip_install.log 2>&1  
D:\quant-claude\env\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 >> pip_install.log 2>&1  
D:\quant-claude\env\python.exe -m pip install pandas==3.0.6 numpy==2.4.6 pyarrow==26.0.0 scipy==1.17.1 structlog psutil duckdb >> pip_install.log 2>&1  
echo DONE >> pip_install.log  
