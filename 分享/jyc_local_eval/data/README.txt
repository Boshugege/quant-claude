<数据占位目录>
本目录不随工具包分发数据。运行上一级目录的 download_data.sh 会把数据下载解压到这里：
  data/cpt_jyc_2026_e2e_bar30m/YYYYMM.0.feather
下载链接：
  https://bigquant-public-download.tos-cn-shanghai.volces.com/jyc2026/cpt_jyc_2026_e2e_bar30m.tar.gz

以下文件同样不分发，程序在本目录下找到时才使用：
  data/exposure/exposure_YYYY.parquet   官方风格暴露表按年保存的副本
  data/exposure/id_map.csv              本地匿名代码与真实代码的对照表（不随仓库发布）
  data/labels_y_off.parquet             用 1 分钟数据重建的标签缓存
缺少 exposure/ 下的两类文件时，本地评估不做官方风格和行业剔除，结果与平台口径不一致。
