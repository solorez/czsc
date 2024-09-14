import sys

sys.path.insert(0, r"D:\github\czsc")
import talib
from czsc.utils import ta
from czsc.connectors import cooperation as coo
import numpy as np



df = coo.get_raw_bars(symbol="SFIC9001", freq="30分钟", fq="后复权", sdt="20100101", edt="20210301", raw_bars=False)


def test_with_numpy():
    df1 = df.copy()
    df1["x"] = ta.LINEARREG_ANGLE(df["close"].values, 10)


def test_with_talib():
    df1 = df.copy()
    df1["x"] = talib.LINEARREG_ANGLE(df["close"].values, 10)


# talib.stream_ACOS()
# # Example usage:
# real_data = np.array([0.5, -0.5, 1.0])
# acos_results = ta.ACOS(real_data)
# print(acos_results)
#
#
#
# talib.stream_ADD()
# # Example usage:
# real_data_0 = np.array([1, 2, 3])
# real_data_1 = np.array([4, 5, 6])
# added_results = ta.ADD(real_data_0, real_data_1)
# print(added_results)
#
#
# talib.stream_AD()
# # Example usage:
# high_data = np.array([100, 102, 101, 103])
# low_data = np.array([98, 99, 98, 100])
# close_data = np.array([99, 101, 100, 102])
# volume_data = np.array([1000, 1500, 1200, 1300])
# ad_results = AD(high_data, low_data, close_data, volume_data)
# print(ad_results)
#
#
# talib.stream_ADOSC()
# # Example usage:
# high = np.array([100, 102, 101, 103])
# low = np.array([98, 99, 98, 100])
# close = np.array([99, 101, 100, 102])
# volume = np.array([1000, 1500, 1200, 1300])
# adosc_results = ADOSC(high_data, low_data, close_data, volume_data)
# print(adosc_results)
#
#
# talib.stream_ADX()
#
# # Example usage:
# high_data = np.array([100, 102, 101, 103])
# low_data = np.array([98, 99, 98, 100])
# close_data = np.array([99, 101, 100, 102])
# adx_results1 = ADX(high_data, low_data, close_data)
# adx_results2 = talib.stream_ADX(high_data, low_data, close_data)
# print(adx_results1)
# print(adx_results2)
#
#
# talib.adxr
# talib.stream_APO()
# talib.stream_AROON()
# talib.stream_ASIN()
# talib.stream_ATAN()
# talib.stream_ATR()
# talib.stream_AVGPRICE()
# talib.stream_BBANDS()
# talib.stream_BETA()
# talib.stream_BOP()
# talib.stream_CCI()
# talib.stream_BBANDS()
# talib.stream_CDL2CROWS()
# talib.stream_CDL3BLACKCROWS()
# talib.stream_CDL3INSIDE()
# talib.stream_CDL3LINESTRIKE()
# talib.stream_CDL3OUTSIDE()
# talib.stream_CDL3STARSINSOUTH()
# talib.stream_CDL3WHITESOLDIERS()
# talib.stream_CDLABANDONEDBABY()
# talib.stream_CDLADVANCEBLOCK()
# talib.stream_CDLBELTHOLD()
# talib.stream_CDLBREAKAWAY()
# # 全局搜索无CDL*调用
# talib.stream_CEIL()
# talib.stream_CMO()
# talib.stream_CORREL()
# talib.stream_COS()
# talib.stream_COSH()
# talib.stream_DEMA()
# talib.stream_DIV()
# talib.stream_DX()
# talib.stream_EXP()
# talib.stream_FLOOR()
# talib.stream_HT_DCPERIOD()
# #全局搜索.HT_*無調用
# talib.stream_KAMA()
# talib.stream_LINEARREG_()
# #全局搜索.LINEARREG_*無調用
# talib.stream_LN()
# talib.stream_LOG10()
# talib.stream_MACDEXT()
# talib.stream_MACDFIX()
# talib.stream_MAMA()
# talib.stream_MAVP()
# talib.stream_MAX()
# talib.stream_MEDPRICE()
# talib.stream_MFI()
# talib.stream_MIDPOINT()
# talib.stream_MIDPRICE()
# talib.stream_MIN()
# talib.stream_MOM()
# talib.stream_MULT()
# talib.stream_NATR()
# talib.nvi
# talib.stream_OBV()
# talib.stream_PLUS_DI()
# talib.stream_PLUS_DM()
# talib.PPO()
# talib.PVI
# talib.ROC()
talib.SAR()
# talib.SQRT()
talib.STDDEV()
talib.STOCH()
talib.SUB()
talibta.SUM()
talib.T3()
talib.Tan()
talib.TEMA()
talib.TRANGE()
talib.TRIMA()
talib.TRIX()
talib.TSF()
talib.TYPPRICE()
talib.ULTOSC()
talib.VAR()
talib.WCLPRICE()
talib.WILLR()
talib.WMA()



# # Example usage:
# real_data = np.array([100, 102, 101, 103])
# upperband, middleband, lowerband = ta.BBANDS(real_data)
# print("Upper Band:", upperband)
# print("Middle Band:", middleband)
# print("Lower Band:", lowerband)





# Example usage:
# real_data = np.array([100.0, 102.0, 101.0,100.0, 102.0, 101.0,100.0, 102.0, 101.0,100.0, 102.0, 101.0,100.0, 102.0, 101.0,100.0, 102.0, 101.0,100.0, 102.0, 101.0,100.0, 102.0, 101.0,100.0, 102.0, 101.0, 103.0])
# ma_result_ta = ta.MA(real_data, timeperiod=30, matype=0)
# ma_result_talib = talib.stream_MA(real_data, timeperiod=30, matype=0)
# print(ma_result_ta)
# print(ma_result_talib)



# Example usage:
high_data = np.array([100, 102, 101, 103])
low_data = np.array([98, 99, 98, 100])
close_data = np.array([99, 101, 100, 102])
plus_di_results =ta.PLUS_DI(high_data, low_data, close_data)
print(plus_di_results)


# 示例数据
high = np.array([100, 102, 101, 105, 107, 106])
low = np.array([99, 98, 99, 103, 104, 105])
close = np.array([100, 101, 100, 104, 106, 105])

# 计算STOCH
slowk, slowd = STOCH(high, low, close)
print("SlowK:", slowk)
print("SlowD:", slowd)