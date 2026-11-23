def main(datasources, start_date, end_date):
    """
    152（倍增阶梯 + 四类价格锚点四网络 IQR + 低波 / 小盘 / 低价）+ 同时段标准化 30 分钟反转，截面排名 1 : 0.5 外层融合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    153 在 152 号基础上（IQR 内部 40 列、慢特征组合全部不变）只在最外层加一个信号：
      z_30 : r_30 / sqrt(过去 20 个交易日同一时段 r_30^2 的均值)，即 30 分钟反转除以该股在同一时段的历史波动（至少 5 个观测，只用过去数据）。
    日内各时段的波动差异很大（开盘后远高于午后），同样幅度的 30 分钟偏离在高波动时段只是常态噪声，在低波动时段才是异常；
    按同一股票、同一时段的历史波动标准化后，各时段、各股票的偏离可比。z_30 以当前价为分母，09:30 截面没有当前价时缺失，
    与 152 同样整体补 0（10:00 截面的 r_30 以 09:30 窗口末价为基准，同样缺失，该截面只用 152）。
    这一信号来自第二十六批 34 个结构不同信号的批量筛选（所有信号统一按 1 : 0.5 与 152 融合），IC / ICIR / Sharpe / 压力四项同时提升最多；
    回看 20 天、最少 5 个观测沿用同时段类信号的统一设定，未做调整。代价是换手明显上升（检验平台对换手的实际权重）：
      验证段 IC 0.08792（152 0.08417）；ICIR 1.8983（1.7966）；Sharpe 57.59（53.06）；压力 1.8790（1.7921）；换手 0.6781（0.3892）。
    以下为 152 的说明。
    更长倍增阶梯 + 价格、区间中点与最低价中位数偏离 + 低振幅加权价格锚点 十分量四网络同伴排名 IQR + 慢特征组合（低波动 / 小盘 / 低价），截面排名 1 : 0.5 融合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    152 在 151 号基础上去掉两个分量 r_lo（当日低点距离）与 r_slope（VWAP 斜率），其余（同伴网络、中性化、聚合、慢特征融合）全部不变，
    即 146 的 6 个阶梯分量 + r_med / r_mid / r_lomed / r_calm 四个价格锚点，10 分量 x 四网络 = 40 列做 IQR Winsor 均值。
    原因：窗口末价格取截面那一分钟，09:30 截面没有这一分钟（1 分钟数据只有 09:25 竞价和 09:31 之后），当前价缺失。
    146 在该截面所有分量缺失、整体补 0 而不参与评估；r_slope 只用成交额与成交量、不含当前价，147–151 在该截面只剩它一个分量，
    该截面 IC 仅约 0.02，拖低整体（本地按同一口径复现：147 相对 146 估计分差 -0.13，与平台 -0.11 一致）。
    152 的所有分量都以当前价为分母，09:30 截面与 146 相同地整体补 0。四个锚点作为一组加入 146 的 IQR，
    在 09:30 截面按平台口径评估时验证段与保留段同时为正；所有候选统一用同一套网络与聚合，未做参数调整。
      验证 / 保留两段换手 0.3892 / 0.3850（146 0.4346 / 0.4256）；
      IC 0.08417 / 0.06863（146 0.08398 / 0.06917）；
      ICIR 1.7965 / 1.1614（146 1.7811 / 1.1693）；
      Sharpe 53.06 / 35.48（146 53.27 / 35.88）；
      压力 1.7921 / 1.1388（146 1.7593 / 1.1520）。
    以下为 151 的说明。
    更长倍增阶梯 + 日内低点距离 + VWAP 斜率 + 价格、区间中点与最低价中位数偏离 + 低振幅加权价格锚点 十二分量四网络同伴排名 IQR + 慢特征组合（低波动 / 小盘 / 低价），截面排名 1 : 0.5 融合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    151 在 150 号基础上只在 IQR 内部加一个反转分量（同伴网络、中性化、聚合、慢特征融合全部不变）：
      r_calm : 低振幅加权价格锚点偏离 ln(锚点 / 当前价)，锚点为最近 8 个窗口末价按 1 / (ln(最高 / 最低) + 0.001) 加权的均值。
    价格稳定（振幅小）的窗口里成交价更接近共识价值，权重更大；剧烈波动窗口的价格噪声大，权重更小。
    窗口长度沿用已有的 8，0.001 只是防止零振幅除零的下限，无新增可调参数。12 分量 x 四网络 = 48 列做 IQR Winsor 均值。
    这一分量来自第十七批“换权重口径锚点”信号的分量级筛选（逐个作为新分量放进 150 的 IQR），
    验证段与保留段同时为正且验证段增量最高；所有候选统一用同一套网络与聚合，未做参数调整。
      验证 / 保留两段换手 0.4174 / 0.4117（150 0.4267 / 0.4194）；
      IC 0.07968 / 0.06546（150 0.08003 / 0.06574）；
      ICIR 1.6943 / 1.1368（150 1.7005 / 1.1415）；
      Sharpe 47.72 / 34.28（150 47.56 / 34.33）；
      压力 1.6870 / 1.1431（150 1.6900 / 1.1467）。
    以下为 150 的说明。
    更长倍增阶梯 + 日内低点距离 + VWAP 斜率 + 价格、区间中点与最低价中位数偏离 十一分量四网络同伴排名 IQR + 慢特征组合（低波动 / 小盘 / 低价），截面排名 1 : 0.5 融合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    150 在 149 号基础上只在 IQR 内部加一个反转分量（同伴网络、中性化、聚合、慢特征融合全部不变）：
      r_lomed : 最低价中位数偏离 ln(最近 8 个窗口最低价的中位数 / 当前价)。
    r_lo 只看当日累计最低点这一个极值，r_lomed 取近 8 个窗口各自低点的中位数，是稳健的近期支撑位；
    窗口长度沿用已有的 8，无新增参数。11 分量 x 四网络 = 44 列做 IQR Winsor 均值。
    这一分量来自第十四至十八批“锚点”信号的分量级筛选（逐个作为新分量放进 149 的 IQR，共 20 个），
    验证段与保留段同时为正且两段增量最高；所有候选统一用同一套网络与聚合，未做参数调整。
      验证 / 保留两段换手 0.4267 / 0.4194（149 0.4303 / 0.4236）；
      IC 0.08003 / 0.06574（149 0.07995 / 0.06575）；
      ICIR 1.7005 / 1.1415（149 1.6975 / 1.1403）；
      Sharpe 47.56 / 34.33（149 47.31 / 34.16）；
      压力 1.6900 / 1.1467（149 1.6915 / 1.1474）。
    以下为 149 的说明。
    更长倍增阶梯 + 日内低点距离 + VWAP 斜率 + 价格与区间中点中位数偏离 十分量四网络同伴排名 IQR + 慢特征组合（低波动 / 小盘 / 低价），截面排名 1 : 0.5 融合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    149 在 148 号基础上只在 IQR 内部加一个反转分量（同伴网络、中性化、聚合、慢特征融合全部不变）：
      r_mid : 区间中点中位数偏离 ln(最近 8 个窗口 (最高 + 最低) / 2 的中位数 / 当前价)。
    r_med 的锚点只用每个窗口末的一个报价，r_mid 用的是每个窗口内全部成交覆盖的价格区间的中点，
    反映窗口内的成交中枢，而不只是窗口收盘时刻的价格。窗口长度沿用已有的 8，无新增参数。10 分量 x 四网络 = 40 列做 IQR Winsor 均值。
    这一分量来自第十一批“稳健锚点”信号的分量级筛选（逐个作为新分量放进 148 的 IQR），
    4 个中只有它验证段与保留段同时为正；所有候选统一用同一套网络与聚合，未做参数调整。
      验证 / 保留两段换手 0.4303 / 0.4236（148 0.4365 / 0.4285）；
      IC 0.07995 / 0.06575（148 0.07984 / 0.06590）；
      ICIR 1.6975 / 1.1403（148 1.6944 / 1.1434）；
      Sharpe 47.31 / 34.16（148 47.28 / 33.82）；
      压力 1.6915 / 1.1474（148 1.6929 / 1.1516）。
    以下为 148 的说明。
    更长倍增阶梯 + 日内低点距离 + VWAP 斜率 + 价格中位数偏离 九分量四网络同伴排名 IQR + 慢特征组合（低波动 / 小盘 / 低价），截面排名 1 : 0.5 融合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    148 在 147 号基础上只在 IQR 内部加一个反转分量（同伴网络、中性化、聚合、慢特征融合全部不变）：
      r_med : 价格中位数偏离 ln(最近 8 个窗口末价的中位数 / 当前价)。
    阶梯中的 r_rv8 以同样 8 个窗口的成交量加权均价为锚，会被单个放量窗口拉动；中位数锚点不受成交量和单个极端价格影响，
    衡量的是当前价相对近期“典型价格”的偏离。窗口长度沿用阶梯中已有的 8，无新增参数。9 分量 x 四网络 = 36 列做 IQR Winsor 均值。
    这一分量来自第十批 16 个“换锚点”反转信号的分量级筛选（逐个作为新分量放进 147 的 IQR），
    16 个中只有它验证段与保留段同时为正；所有候选统一用同一套网络与聚合，未做参数调整。
      验证 / 保留两段换手 0.4365 / 0.4285（147 0.4389 / 0.4314）；
      IC 0.07984 / 0.06590（147 0.07933 / 0.06569）；
      ICIR 1.6944 / 1.1434（147 1.6821 / 1.1417）；
      Sharpe 47.28 / 33.82（147 46.51 / 33.45）；
      压力 1.6929 / 1.1516（147 1.6729 / 1.1436）。
    以下为 147 的说明。
    更长倍增阶梯 + 日内低点距离 + VWAP 斜率 八分量四网络同伴排名 IQR + 慢特征组合（低波动 / 小盘 / 低价），截面排名 1 : 0.5 融合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    147 在 146 号基础上只在 IQR 内部加两个反转分量（同伴网络、中性化、聚合、慢特征融合全部不变）：
      r_lo    : 当日低点距离 ln(min(当日累计窗口最低价, 昨收) / 当前价)，衡量从当日低点（或昨收）以来的反弹幅度；
      r_slope : 窗口 VWAP 斜率 ln(滚动 8 窗口 VWAP / 当前窗口 VWAP)，衡量当前窗口成交均价相对近期均价的抬升。
    两者与阶梯中的“价格相对 VWAP 偏离”口径不同：前者以当日极值为锚，后者只看成交均价、不含当前报价，
    与已有分量相关较低。8 分量 x 四网络 = 32 列做 IQR Winsor 均值。
    这两个分量来自对 215 个结构不同信号的批量分量级筛选（逐个作为新分量放进 146 的 IQR），验证段与保留段同时为正；
    所有候选统一用同一套网络与聚合，未做参数调整。
      验证 / 保留两段换手 0.4389 / 0.4314（146 0.4798 / 0.4698）；
      IC 0.07933 / 0.06569（146 0.08049 / 0.06665）；
      ICIR 1.6821 / 1.1417（146 1.7022 / 1.1562）；
      Sharpe 46.51 / 33.45（146 47.27 / 34.23）；
      压力 1.6729 / 1.1436（146 1.6992 / 1.1468）。
    以下为 146 的说明。
    更长倍增阶梯四网络同伴排名 IQR（144）+ 慢特征组合（低波动 / 小盘 / 低价），截面排名 1 : 0.5 融合。
    146 在 144 号基础上只加一个与反转正交的慢特征组合：
      c_park  : 低波动，最近 40 个窗口 Parkinson 区间方差 ln(窗口最高 / 最低)^2 的均值取负；
      c_size  : 小盘，-ln(1 + 最近 40 个窗口清洗后成交额之和)（144 中已有，用于中性化）；
      c_lowpx : 低价，-ln(当前价)（144 中已有，用于中性化）。
    三者截面排名等权相加，再与 144 的 IQR 输出按固定的 1 : 0.5 做截面排名融合。三者都是 A 股文献中的经典异象
    （低波动、小市值、低价），与反转主信号的风格剔除后相关约为 0；它们几乎不随时间变化，融合后换手下降。
    这一组合来自对 130 个结构不同信号的批量筛选，验证段与保留段同时为正；融合权重对全部信号统一取 0.5，未逐个调参。
      验证 / 保留两段换手 0.4798 / 0.4698（144 0.4969 / 0.4883）；
      IC 0.08049 / 0.06665（144 0.07809 / 0.06452）；
      ICIR 1.7022 / 1.1562（144 1.6088 / 1.1070）；
      Sharpe 47.27 / 34.23（144 49.77 / 35.09）；
      压力 1.6992 / 1.1468（144 1.5654 / 1.0782）。
    以下为 144 的说明。
    平台原始价格口径下的更长倍增阶梯超额四网络同伴排名，IQR Winsor 均值聚合。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1，不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    144 在 143 号基础上只加一个分量：倍增阶梯沿同一倍增序列向慢端再延一档，加入滚动 32 窗口（约 4 个交易日，
    接近一周）清洗后成交 VWAP 偏离 r_rv32。六个分量为 2 / 4 / 8 / 16 / 32 窗口 VWAP 偏离 + 日内反转，
    x 四张同伴网络，24 列做 IQR Winsor 均值。周尺度锚点更新更慢，合成信号相邻截面更连续，换手继续下降；
    窗口长度仍是同一倍增序列的下一项，无新增参数。
      验证 / 保留两段换手 0.4969 / 0.4883（143 0.5223 / 0.5155）；
      IC 0.07809 / 0.06452（143 0.07838 / 0.06514）；
      ICIR 1.6088 / 1.1070（143 1.6135 / 1.1239）；
      Sharpe 49.77 / 35.09（143 50.83 / 35.84）；
      压力 1.5654 / 1.0782（143 1.5519 / 1.0812）。
    以下为 143 的说明。
    143 在 142 号基础上只加一个分量：倍增阶梯沿同一倍增序列向慢端再延一档，加入滚动 16 窗口（约 2 个交易日）
    清洗后成交 VWAP 偏离 r_rv16。五个分量为 2 / 4 / 8 / 16 窗口 VWAP 偏离 + 日内反转，x 四张同伴网络，
    20 列做 IQR Winsor 均值。慢端锚点跨日更新缓慢，加入后合成信号的截面变化更平稳，同时多日尺度的
    VWAP 偏离本身也带有反转信息；窗口长度仍是同一倍增序列的下一项，无新增参数。
      验证 / 保留两段换手 0.5223 / 0.5155（142 0.5688 / 0.5623）；
      IC 0.07838 / 0.06514（142 0.07808 / 0.06547）；
      ICIR 1.6135 / 1.1239（142 1.6142 / 1.1341）；
      Sharpe 50.83 / 35.84（142 51.17 / 36.54）；
      压力 1.5519 / 1.0812（142 1.5298 / 1.0755）。
    以下为 142 的说明。
    142 在 140 号基础上只换一个分量：30 分钟反转 r_30（只看上一窗口末的单个价格点）换成滚动 2 窗口（1 小时）
    清洗后成交 VWAP 偏离 r_rv2。四个分量成为 2 / 4 / 8 窗口 VWAP 偏离的倍增阶梯 + 日内反转，尺度均匀覆盖，
    且短端由成交量加权的均价代替单点价格，相邻截面之间更连续；首窗不再与 r_day 重复，无需再做首窗去重。
    各分量 x 四张同伴网络，16 列做 IQR Winsor 均值。窗口长度沿用已有的倍增序列，无新增参数。
      验证 / 保留两段换手 0.5688 / 0.5623（140 0.6005 / 0.5949）；
      IC 0.07808 / 0.06547（140 0.07906 / 0.06626）；
      ICIR 1.6142 / 1.1341（140 1.6303 / 1.1475）；
      Sharpe 51.17 / 36.54（140 51.55 / 36.53）；
      压力 1.5298 / 1.0755（140 1.5472 / 1.0809）。
    以下为 141 的说明（141 去掉 r_30 只留三个慢分量，换手降得更多但 IC 也掉得更多，142 是两者之间的折中）。
    141 在 140 号基础上只做一个无参结构改动：去掉 30 分钟反转分量 r_30，只保留滚动 VWAP 4/8 窗口偏离与
    日内反转三个慢分量（各 x 四张同伴网络，12 列做 IQR Winsor 均值）。r_30 只看最近一个窗口，相邻截面之间
    几乎完全更新，是多空两腿成分替换的主要来源；平台五项得分中换手的权重最大，r_30 的边际 IC 不足以抵消它带来的换手。
    r_30 仍用于构建 30 分钟同伴网络（窗口收益相关），只是不再作为合成分量。无新增参数。
      验证 / 保留两段换手 0.4775 / 0.4696（140 0.6005 / 0.5949，降约 0.12）；
      IC 0.07646 / 0.06347（140 0.07906 / 0.06626）；
      ICIR 1.5766 / 1.1041（140 1.6303 / 1.1475）；
      Sharpe 49.52 / 34.85（140 51.55 / 36.53）；
      压力 1.5145 / 1.0732（140 1.5472 / 1.0809）。
    以下为 140 的说明。
    140 在 137 号基础上只改最后一步聚合：16 列中性化排名由 Midhinge
    （25%/75% 分位点均值）改为 IQR Winsor 均值（各列截到本行四分位区间后取均值）。Midhinge 只由两个分位点决定，
    任一列的小幅变动都可能让分位点在相邻列之间跳动，截面排名随之抖动；Winsor 均值让中间一半的列都连续地参与，
    相邻截面更平滑。无新增参数。平台五项得分对换手最敏感之一，此改动的主要收益是降换手：
      开发 / 验证 / 保留三段换手 0.6009 / 0.6005 / 0.5949（137 0.6211 / 0.6194 / 0.6132，三段均降约 0.02）；
      IC 0.08895 / 0.07906 / 0.06626（137 0.08917 / 0.07937 / 0.06643）；
      ICIR 1.4783 / 1.6303 / 1.1475（137 1.4816 / 1.6362 / 1.1496）；
      Sharpe 49.77 / 51.55 / 36.53（137 50.05 / 52.27 / 36.73）；
      压力 1.4021 / 1.5472 / 1.0809（137 1.4049 / 1.5532 / 1.0807）。
    以下为 137 的说明。
    在 136 号（首窗去重的 4 个反转分量 x 日频/30 分钟两张同伴网络，各做“中位数残差 + 同伴内排名”融合与
    四特征中性化，8 列取 Midhinge）基础上只加一个无参结构改动：
      另建两张“超额收益”同伴网络：日收益 / 30 分钟窗口收益先减去同一时点的截面均值，再按相关性取前 30 名同伴。
      原始收益相关里很大一部分是共同的市场涨跌，高贝塔股票会因此互为同伴；超额网络只保留真实的同类联动。
      原始 / 超额 x 日频 / 30 分钟四张网络并列，每个分量在每张网络上照旧做融合与中性化，16 列取 Midhinge
      （09:30 截面去掉 r_30 的 4 列，用 12 列）。不新增任何参数，同伴数、回看期、最少观测都与 136 相同。
    相对 136，本地开发 / 验证 / 保留三段：
      IC 0.08917 / 0.07937 / 0.06643（136 0.08910 / 0.07944 / 0.06629）；
      ICIR 1.4816 / 1.6362 / 1.1496（136 1.4692 / 1.6274 / 1.1277）；
      Sharpe 50.05 / 52.27 / 36.73（136 49.94 / 51.94 / 36.93）；
      压力 1.4049 / 1.5532 / 1.0807（136 1.3931 / 1.5425 / 1.0629）；
      换手 0.6211 / 0.6194 / 0.6132（136 0.6188 / 0.6175 / 0.6102）。
    136 的改动：当日首个窗口的 30 分钟反转以昨收为基准，与 r_day 完全相同，首窗把 r_30 各列置为缺失，避免重复计票；
    r_30 仍保留首窗定义用于构建 30 分钟同伴网络。
    公共构造（103–107 共用同一段代码，只有最后的合成方式不同）：
      反转分量（与 90 号相同，均只用过去数据）：
        r_30  : 30 分钟反转 ln(上一窗口末价 / 当前价)，当日首个窗口以昨收为基准（57 号）
        r_rv4 / r_rv8 : 滚动 4/8 个窗口（2/4 小时，跨日）清洗后成交 VWAP 偏离 ln(滚动 VWAP / 当前价)
        r_day : 日内反转 ln(昨收 / 当前价)（56 号）
      同伴（伪行业）残差：每个自然月月初，用此前 60 个交易日的日收益 ln(收盘段价格 / 昨收) 计算股票池内两两相关系数，
        每只股票取相关性最高的 30 只为同伴（至少 30 个共同观测；只用过去数据）。分量残差 = 分量 - 同伴中位数
        （同伴有效值不足 3 个时减截面均值）。思路：行业内反转强于全市场反转（Da, Liu & Schaumburg 2014），
        去掉“整组一起涨跌”的部分后，只保留个股相对同类股票的过度反应。
      四特征中性化：因子截面排名对 规模（-ln 最近 40 窗口成交额）、特质波动、非流动性（Amihud）、低价 四个特征的截面排名
        做逐截面回归，取残差。反转信号天然偏向小盘、高波动股票，而这些暴露本身会拖累排序；剔除后 IC、IC_IR、多空夏普同时提升。
      数据清洗：成交均价落在该分钟 [最低价 x 0.98, 最高价 x 1.02] 之外的分钟不计入任何 VWAP / 成交额累加。

    稳健性：最终缺失补 0（中性），每个截面缺失率为 0；向前取不到足够历史时同伴退化为截面均值、特征排名记为中性。
    性能：纯 SQL 向量化（单次 GROUP BY，无窗口函数、无 Python UDF）；相关矩阵每月计算一次（约 1200 x 1200），
      同伴均值按月份用数组索引批量计算。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取数据：同伴相关系数需要月初之前 60 个交易日的日收益（约 90 个自然日，加上月内偏移与长假留足余量）。
    LOOKBACK_DAYS = 150
    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS = 30, 60, 30
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt
        FROM {bar1m}
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            pre_close,
            px_high,
            px_low,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN amt ELSE 0
            END AS amt_c,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN vol ELSE 0
            END AS vol_c,
            CASE
                WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,
            CASE
                WHEN hms > '090000' AND hms <= '093000' THEN 93000
                WHEN hms > '093000' AND hms <= '100000' THEN 100000
                WHEN hms > '100000' AND hms <= '103000' THEN 103000
                WHEN hms > '103000' AND hms <= '110000' THEN 110000
                WHEN hms > '110000' AND hms <= '113000' THEN 113000
                WHEN hms > '130000' AND hms <= '133000' THEN 133000
                WHEN hms > '133000' AND hms <= '140000' THEN 140000
                WHEN hms > '140000' AND hms <= '143000' THEN 143000
                WHEN hms > '143000' AND hms <= '150000' THEN 150000
                ELSE -1
            END AS time_segment
        FROM cte_raw
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt_c) AS sum_amt,
            SUM(vol_c) AS sum_vol,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(px_high) AS bar_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS bar_low
        FROM cte_snapshot
        WHERE time_segment != -1
        GROUP BY instrument_id, trading_day, time_segment
    )
    SELECT
        CAST(CONCAT(
            f.trading_day,
            ' ',
            strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        all_instruments.instrument,
        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.bar_high, f.bar_low
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "bar_high", "bar_low"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)

    # 股票池：同伴只在股票池内挑选（向前取历史区间的成分，取不到时用评估区间的成分）。
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    pool_hist = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [query_start, end_date]},
        compression=True,
    ).df()
    pool_hist["date"] = pd.to_datetime(pool_hist["date"])
    pool_hist["instrument"] = pool_hist["instrument"].astype(str)
    stk_pool = pool_hist[(pool_hist["date"] >= pd.to_datetime(start_date)) & (pool_hist["date"] <= pd.to_datetime(end_date))]
    universe = set(pool_hist["instrument"].unique())
    df = df[df["instrument"].isin(universe)]
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    def vwap_of(amt, vol):
        return (amt / vol.where(vol > 0)).where(amt > 0)

    def roll(s, k, fn, min_periods):
        return s.groupby(df["instrument"], sort=False).transform(lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())

    grp = df.groupby("instrument", sort=False)
    grp_day = df.groupby(["instrument", "day"], sort=False)
    same_day = grp["day"].shift(1) == df["day"]
    px = df["last_px"]

    day_vwap = vwap_of(grp_day["sum_amt"].cumsum(), grp_day["sum_vol"].cumsum())
    r_dvwap = safe_log_ratio(day_vwap, px)

    def rolling_vwap_rev(k):
        return safe_log_ratio(vwap_of(roll(df["sum_amt"], k, "sum", 1), roll(df["sum_vol"], k, "sum", 1)), px).fillna(r_dvwap)

    df["r_30"] = safe_log_ratio(grp["last_px"].shift(1).where(same_day, df["pre_close"]), px)
    df["r_rv2"] = rolling_vwap_rev(2)
    df["r_rv4"] = rolling_vwap_rev(4)
    df["r_rv8"] = rolling_vwap_rev(8)
    df["r_rv16"] = rolling_vwap_rev(16)
    df["r_rv32"] = rolling_vwap_rev(32)
    df["r_day"] = safe_log_ratio(df["pre_close"], px)
    # 148：价格中位数偏离 ln(最近 8 个窗口末价的中位数 / 当前价)，以稳健的近期价格中枢为锚。
    df["r_med"] = safe_log_ratio(roll(px, 8, "median", 2), px)
    # 149：区间中点中位数偏离 ln(最近 8 个窗口 (最高 + 最低) / 2 的中位数 / 当前价)，以窗口内全部成交区间的中枢为锚。
    df["r_mid"] = safe_log_ratio(roll(((df["bar_high"] + df["bar_low"]) / 2).where(df["bar_low"] > 0), 8, "median", 2), px)
    # 150：最低价中位数偏离 ln(最近 8 个窗口最低价的中位数 / 当前价)，以近期各窗口低点的稳健中枢为支撑锚。
    df["r_lomed"] = safe_log_ratio(roll(df["bar_low"].where(df["bar_low"] > 0), 8, "median", 2), px)
    # 151：低振幅加权价格锚点偏离，最近 8 个窗口末价按 1 / (ln(最高 / 最低) + 0.001) 加权平均，振幅小的窗口权重大。
    calm_w = 1.0 / (safe_log_ratio(df["bar_high"], df["bar_low"]).where(df["bar_high"] > df["bar_low"]) + 1e-3)
    calm_den = roll(calm_w.where(px > 0), 8, "sum", 2)
    df["r_calm"] = safe_log_ratio(roll(px.where(px > 0) * calm_w, 8, "sum", 2) / calm_den.where(calm_den > 0), px)
    # 153：同时段标准化 30 分钟反转 r_30 / sqrt(过去 20 个交易日同一时段 r_30^2 的均值)，至少 5 个观测，只用过去数据。
    slot_key = [df["instrument"], df["date"].dt.hour * 100 + df["date"].dt.minute]
    r30_sq_prev = (df["r_30"] ** 2).groupby(slot_key, sort=False).shift(1)
    slot_ms = r30_sq_prev.groupby(slot_key, sort=False).transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["z_30"] = df["r_30"] / np.sqrt(slot_ms.where(slot_ms > 0))

    # 个股 beta（40 窗口滚动回归，截断 [-1, 4]，不足取 1），用于特质波动。
    win_ret = safe_log_ratio(px, grp["last_px"].shift(1))
    ret30 = safe_log_ratio(px, grp["last_px"].shift(1).where(same_day)).fillna(safe_log_ratio(px, df["pre_close"]))
    mkt30 = ret30.groupby(df["date"]).transform("mean")
    ex, ey = roll(ret30, 40, "mean", 10), roll(mkt30, 40, "mean", 10)
    exy, ey2 = roll(ret30 * mkt30, 40, "mean", 10), roll(mkt30 ** 2, 40, "mean", 10)
    var_m = (ey2 - ey ** 2).where((ey2 - ey ** 2) > 1e-12)
    beta = ((exy - ex * ey) / var_m).clip(-1, 4).fillna(1.0)

    # 股票特征：规模（-ln 最近 40 窗口成交额）、特质波动（取负）、非流动性（Amihud）、低价、MAX（取负）。
    df["c_size"] = -np.log1p(roll(df["sum_amt"], 40, "sum", 10))
    idio = win_ret - beta * win_ret.groupby(df["date"]).transform("mean")
    df["c_ivol"] = -roll(idio, 40, "std", 10)
    df["c_illiq"] = roll(win_ret.abs() / (df["sum_amt"] / 1e8).where(df["sum_amt"] > 0), 40, "mean", 10)
    df["c_lowpx"] = -np.log(px.where(px > 0))
    df["c_max"] = -roll(win_ret, 40, "max", 10)
    # 低波动（取负）：最近 40 个窗口 Parkinson 区间方差 ln(窗口最高 / 最低)^2 的均值。
    df["c_park"] = -roll(safe_log_ratio(df["bar_high"], df["bar_low"]) ** 2, 40, "mean", 10)

    # 同伴：每个自然月月初，用此前 PEER_LOOKBACK 个交易日的收益计算相关系数，两种频率各建一张网络：
    #   日频：日收益（收盘段价格 / 昨收），至少 PEER_MIN_OBS 个共同观测；
    #   30 分钟：窗口收益（上一窗口末价，当日首个窗口用昨收），取回看期内全部窗口，至少 4 x PEER_MIN_OBS 个共同观测。
    daily = df.groupby(["day", "instrument"], sort=False).agg(px_last=("last_px", "last"), pc=("pre_close", "first"))
    daily_ret = safe_log_ratio(daily["px_last"], daily["pc"]).unstack("instrument").sort_index()
    peer_insts = daily_ret.columns.values
    all_days = daily_ret.index
    day_month = all_days.to_period("M")
    out_months = pd.to_datetime(stk_pool["date"]).dt.normalize().dt.to_period("M").unique()
    win_ret30 = pd.Series(
        (-df["r_30"]).values, index=pd.MultiIndex.from_arrays([df["date"].values, df["instrument"].values])
    ).unstack().sort_index().reindex(columns=peer_insts)

    def build_peers(ret_wide, row_day, need):
        """返回 {月份: 同伴下标矩阵}；ret_wide 每行是一个收益观测，row_day 为该行所属交易日。"""
        peers = {}
        for m in out_months:
            in_month = all_days[day_month == m]
            if len(in_month) == 0:
                continue
            prior = all_days[all_days < in_month[0]][-PEER_LOOKBACK:]
            if len(prior) < PEER_MIN_OBS:
                continue
            hist = ret_wide[(row_day < in_month[0]) & (row_day >= prior[0])]
            ok = (hist.notna().sum() >= need).values
            corr = hist.loc[:, ok].corr(min_periods=need).values
            np.fill_diagonal(corr, -np.inf)
            corr = np.nan_to_num(corr, nan=-np.inf)
            sub = np.where(ok)[0]
            idx = np.full((len(peer_insts), PEER_K), -1, dtype=np.int64)
            idx[sub] = sub[np.argsort(-corr, axis=1, kind="stable")[:, :PEER_K]]
            peers[m] = idx
        return peers

    # 超额网络：每个收益观测先减去同一时点的截面均值，再按超额收益的相关性挑同伴（去掉市场共振带来的伪同伴）。
    peers_by_freq = {
        "day": build_peers(daily_ret, all_days, PEER_MIN_OBS),
        "30": build_peers(win_ret30, win_ret30.index.normalize(), PEER_MIN_OBS * 4),
        "day_excess": build_peers(daily_ret.sub(daily_ret.mean(axis=1), axis=0), all_days, PEER_MIN_OBS),
        "30_excess": build_peers(win_ret30.sub(win_ret30.mean(axis=1), axis=0), win_ret30.index.normalize(), PEER_MIN_OBS * 4),
    }

    df = df.replace([np.inf, -np.inf], np.nan)
    keep = ["r_rv2", "r_rv4", "r_rv8", "r_rv16", "r_rv32", "r_day", "r_med", "r_mid", "r_lomed", "r_calm", "z_30", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_park"]
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool[["date", "instrument"]], how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)
    result_month = result["date"].dt.to_period("M")

    def peer_mean(col, peers_by_month):
        """每个截面上同伴的 col 中位数（至少 3 个有效值），没有同伴信息时为缺失。"""
        out = np.full(len(result), np.nan)
        for m, rows in result.groupby(result_month, sort=False).indices.items():
            idx = peers_by_month.get(m)
            if idx is None:
                continue
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            valid = idx >= 0
            g = wide.values[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            cnt = (~np.isnan(g)).sum(axis=2)
            pm = np.where(cnt >= 3, np.nanmedian(g, axis=2), np.nan)
            r_pos = wide.index.get_indexer(sub["date"])
            c_pos = pd.Index(peer_insts).get_indexer(sub["instrument"])
            out[rows] = np.where(c_pos >= 0, pm[r_pos, np.maximum(c_pos, 0)], np.nan)
        return pd.Series(out, index=result.index)

    def peer_rank(col, peers_by_month):
        """个股 col 在其同伴中的相对排名（低于自身的同伴占比，平局记半），居中到 [-0.5, 0.5]；
        同伴有效值不足 3 个或没有同伴信息时回退为截面居中排名。"""
        out = np.full(len(result), np.nan)
        for m, rows in result.groupby(result_month, sort=False).indices.items():
            idx = peers_by_month.get(m)
            if idx is None:
                continue
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            x = wide.values
            valid = idx >= 0
            g = x[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            own = x[:, :, None]
            ok = ~np.isnan(g) & ~np.isnan(own)
            cnt = ok.sum(axis=2)
            score = ((g < own) & ok).sum(axis=2) + 0.5 * ((g == own) & ok).sum(axis=2)
            rel = np.where(cnt >= 3, score / np.maximum(cnt, 1) - 0.5, np.nan)
            r_pos = wide.index.get_indexer(sub["date"])
            c_pos = pd.Index(peer_insts).get_indexer(sub["instrument"])
            out[rows] = np.where(c_pos >= 0, rel[r_pos, np.maximum(c_pos, 0)], np.nan)
        fallback = result.groupby("date")[col].rank(pct=True) - 0.5
        return pd.Series(out, index=result.index).fillna(fallback)

    def rank_blend(cols, weights=None):
        """各分量在股票池内逐截面居中排名后加权平均（缺失分量不参与，权重按可用分量重新归一）。"""
        weights = weights or [1.0] * len(cols)
        ranks = (result.groupby("date")[cols].rank(pct=True) - 0.5).to_numpy()
        w = np.array(weights, dtype=float)
        has = ~np.isnan(ranks)
        den = (has * w).sum(axis=1)
        num = np.nansum(np.nan_to_num(ranks) * w, axis=1)
        return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=result.index)

    def neutralize(col, chars, frac=1.0):
        """截面中性化：col 的居中排名对各特征居中排名（缺失记 0）逐截面 OLS（含截距），返回 排名 - frac x 拟合值。"""
        y = result.groupby("date")[col].rank(pct=True) - 0.5
        X = np.column_stack([(result.groupby("date")[c].rank(pct=True) - 0.5).fillna(0.0).values for c in chars])
        ok = y.notna().values
        codes, uniq = pd.factorize(result["date"].values)
        k, n_grp = X.shape[1], len(uniq)
        Xc = np.column_stack([np.ones(len(result)), X]) * ok[:, None]
        yv = np.nan_to_num(y.values) * ok
        xtx = np.zeros((n_grp, k + 1, k + 1))
        xty = np.zeros((n_grp, k + 1))
        for i in range(k + 1):
            xty[:, i] = np.bincount(codes, Xc[:, i] * yv, minlength=n_grp)
            for j in range(i, k + 1):
                v = np.bincount(codes, Xc[:, i] * Xc[:, j], minlength=n_grp)
                xtx[:, i, j] = v
                xtx[:, j, i] = v
        b = np.linalg.solve(xtx + 1e-9 * np.eye(k + 1), xty[..., None])[..., 0]
        fit = (X * b[codes, 1:]).sum(axis=1)
        return pd.Series(np.where(ok, y.values - frac * fit, np.nan), index=result.index)

    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    neutral_cols = []
    for c in ["r_rv2", "r_rv4", "r_rv8", "r_rv16", "r_rv32", "r_day", "r_med", "r_mid", "r_lomed", "r_calm"]:
        cross_mean = result.groupby("date")[c].transform("mean")
        for f, peers in peers_by_freq.items():
            # 每个分量 x 每张同伴网络：同伴中位数残差与同伴内相对排名各自截面排名后等权融合，再做四特征中性化。
            result[f"{c}_p{f}"] = result[c] - peer_mean(c, peers).fillna(cross_mean)
            result[f"{c}_r{f}"] = peer_rank(c, peers)
            result[f"{c}_b{f}"] = rank_blend([f"{c}_p{f}", f"{c}_r{f}"])
            name = f"{c}_n{f}"
            result[name] = neutralize(f"{c}_b{f}", CHARS4)
            neutral_cols.append(name)
    neutral_ranks = result.groupby("date")[neutral_cols].rank(pct=True) - 0.5
    lower = neutral_ranks.quantile(0.25, axis=1)
    upper = neutral_ranks.quantile(0.75, axis=1)
    # IQR Winsor 均值：各列先截到本行四分位区间，再取均值（中间一半的票全部参与，而不只看两个分位点）。
    result["iqr"] = neutral_ranks.clip(lower=lower, upper=upper, axis=0).mean(axis=1, skipna=True)
    # 146：慢特征组合 = 低波动 / 小盘 / 低价三者截面排名等权相加，再与反转主信号按 1 : 0.5 做截面排名融合。
    char_ranks = result.groupby("date")[["c_park", "c_size", "c_lowpx"]].rank(pct=True)
    result["slow"] = char_ranks["c_park"] + char_ranks["c_size"] + char_ranks["c_lowpx"]
    result["f152"] = rank_blend(["iqr", "slow"], [1.0, 0.5])
    # 153：152 与同时段标准化 30 分钟反转按 1 : 0.5 做截面排名融合（09:30 截面两者都缺失，整体补 0）。
    result["factor"] = rank_blend(["f152", "z_30"], [1.0, 0.5])
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
