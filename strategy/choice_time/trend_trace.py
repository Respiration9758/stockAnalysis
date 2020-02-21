import numpy as np
import pandas as pd
import tushare as ts
import tools.dataProcess
def trend_trace_MA(scode, startTime, endTime, snum, lnum):
    """
    根据股票代码，选择一定时间范围内的历史交易数据，根据给出的短期和长期均线的时间间
    隔计算均线，最终返回相关历史数据以及买卖点。

    :param scode: 股票代码
    :param startTime: 历史数据的开始时间
    :param endTime: 历史数据的结束时间
    :param snum: 短期均线长度
    :param lnum: 长期均线长度
    :return: shdata： 代表历史行情数据
    :return: bs_points： 代表判断的买卖点

    """

    shdata = ts.get_k_data(scode, startTime, endTime)[['date', 'open', 'close', 'high', 'low', 'volume']]
    shdata.set_index('date', inplace=True)
    tools.dataProcess.calculateMA(shdata, snum)
    tools.dataProcess.calculateMA(shdata, lnum)

    golden_cross = []
    death_cross = []

    # 求金叉死叉的方法二
    ser1 = shdata['MA_'+str(snum)] < shdata['MA_'+str(lnum)]
    ser2 = shdata['MA_'+str(snum)] >= shdata['MA_'+str(lnum)]
    # One-dimensional ndarray with axis labels (including time series).
    # print(shdata.info())
    #
    death_cross = shdata[ser1 & ser2.shift(1)].index
    golden_cross = shdata[-(ser1 | ser2.shift(1))].index

    ser1 = pd.Series(1, index=golden_cross)
    ser2 = pd.Series(0, index=death_cross)
    bs_points = ser1.append(ser2).sort_index()
    print(len(golden_cross))
    print(len(death_cross))
    # print(len(bs_points))

    return shdata[['open', 'close', 'high', 'low', 'volume']], bs_points


shdata, bs_points = trend_trace_MA('600519','1988-01-01','2020-01-21',5, 20)
print(len(bs_points))
shdata['label'] = bs_points
print(shdata.shape)
print(shdata.info())
shdata.dropna(axis=0, how='any', inplace=True)
print(shdata.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

shdata = MinMaxScaler().fit_transform(shdata)
# x_train, x_test, y_train, y_test = train_test_split(shdata.iloc[:,:-1]
#                                                     , shdata.iloc[:,-1], test_size=0.2)

rfc = RandomForestClassifier(n_estimators=20,random_state=90)
score_pre = cross_val_score(rfc, shdata[:, :-1]
                            , shdata[:, -1], cv=10).mean()
print(score_pre)