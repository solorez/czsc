# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2022/1/24 15:01
describe: 常用技术分析指标
"""
import numpy as np
import pandas as pd


def SMA(close: np.array, timeperiod=5):
    """简单移动平均

    https://baike.baidu.com/item/%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E7%BA%BF/217887

    :param close: np.array
        收盘价序列
    :param timeperiod: int
        均线参数
    :return: np.array
    """
    res = []
    for i in range(len(close)):
        if i < timeperiod:
            seq = close[0 : i + 1]
        else:
            seq = close[i - timeperiod + 1 : i + 1]
        res.append(seq.mean())
    return np.array(res, dtype=np.double).round(4)


def EMA(close: np.array, timeperiod=5):
    """
    https://baike.baidu.com/item/EMA/12646151

    :param close: np.array
        收盘价序列
    :param timeperiod: int
        均线参数
    :return: np.array
    """
    res = []
    for i in range(len(close)):
        if i < 1:
            res.append(close[i])
        else:
            ema = (2 * close[i] + res[i - 1] * (timeperiod - 1)) / (timeperiod + 1)
            res.append(ema)
    return np.array(res, dtype=np.double).round(4)


def MACD(close: np.array, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD 异同移动平均线
    https://baike.baidu.com/item/MACD%E6%8C%87%E6%A0%87/6271283

    :param close: np.array
        收盘价序列
    :param fastperiod: int
        快周期，默认值 12
    :param slowperiod: int
        慢周期，默认值 26
    :param signalperiod: int
        信号周期，默认值 9
    :return: (np.array, np.array, np.array)
        diff, dea, macd
    """
    ema12 = EMA(close, timeperiod=fastperiod)
    ema26 = EMA(close, timeperiod=slowperiod)
    diff = ema12 - ema26
    dea = EMA(diff, timeperiod=signalperiod)
    macd = (diff - dea) * 2
    return diff.round(4), dea.round(4), macd.round(4)


def KDJ(close: np.array, high: np.array, low: np.array):
    """

    :param close: 收盘价序列
    :param high: 最高价序列
    :param low: 最低价序列
    :return:
    """
    n = 9
    hv = []
    lv = []
    for i in range(len(close)):
        if i < n:
            h_ = high[0 : i + 1]
            l_ = low[0 : i + 1]
        else:
            h_ = high[i - n + 1 : i + 1]
            l_ = low[i - n + 1 : i + 1]
        hv.append(max(h_))
        lv.append(min(l_))

    hv = np.around(hv, decimals=2)
    lv = np.around(lv, decimals=2)
    rsv = np.where(hv == lv, 0, (close - lv) / (hv - lv) * 100)

    k = []
    d = []
    j = []
    for i in range(len(rsv)):
        if i < n:
            k_ = rsv[i]
            d_ = k_
        else:
            k_ = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
            d_ = (2 / 3) * d[i - 1] + (1 / 3) * k_

        k.append(k_)
        d.append(d_)
        j.append(3 * k_ - 2 * d_)

    k = np.array(k, dtype=np.double)
    d = np.array(d, dtype=np.double)
    j = np.array(j, dtype=np.double)
    return k.round(4), d.round(4), j.round(4)


def RSQ(close: [np.array, list]) -> float:
    """拟合优度 R Square

    :param close: 收盘价序列
    :return:
    """
    x = list(range(len(close)))
    y = np.array(close)
    x_squred_sum = sum([x1 * x1 for x1 in x])
    xy_product_sum = sum([x[i] * y[i] for i in range(len(x))])
    num = len(x)
    x_sum = sum(x)
    y_sum = sum(y)
    delta = float(num * x_squred_sum - x_sum * x_sum)
    if delta == 0:
        return 0
    y_intercept = (1 / delta) * (x_squred_sum * y_sum - x_sum * xy_product_sum)
    slope = (1 / delta) * (num * xy_product_sum - x_sum * y_sum)

    y_mean = np.mean(y)
    ss_tot = sum([(y1 - y_mean) * (y1 - y_mean) for y1 in y]) + 0.00001
    ss_err = sum([(y[i] - slope * x[i] - y_intercept) * (y[i] - slope * x[i] - y_intercept) for i in range(len(x))])
    rsq = 1 - ss_err / ss_tot

    return round(rsq, 4)


def plus_di(high, low, close, timeperiod=14):
    """
    Calculate Plus Directional Indicator (PLUS_DI) manually.

    Parameters:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    close (pd.Series): Closing price series.
    timeperiod (int): Number of periods to consider for the calculation.

    Returns:
    pd.Series: Plus Directional Indicator values.
    """
    # Calculate the +DM (Directional Movement)
    dm_plus = high - high.shift(1)
    dm_plus[dm_plus < 0] = 0  # Only positive differences are considered

    # Calculate the True Range (TR)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

    # Smooth the +DM and TR with Wilder's smoothing method
    smooth_dm_plus = dm_plus.rolling(window=timeperiod).sum()
    smooth_tr = tr.rolling(window=timeperiod).sum()

    # Avoid division by zero
    smooth_tr[smooth_tr == 0] = np.nan

    # Calculate the Directional Indicator
    plus_di_ = 100 * (smooth_dm_plus / smooth_tr)

    return plus_di_


def minus_di(high, low, close, timeperiod=14):
    """
    Calculate Minus Directional Indicator (MINUS_DI) manually.

    Parameters:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    close (pd.Series): Closing price series.
    timeperiod (int): Number of periods to consider for the calculation.

    Returns:
    pd.Series: Minus Directional Indicator values.
    """
    # Calculate the -DM (Directional Movement)
    dm_minus = (low.shift(1) - low).where((low.shift(1) - low) > (high - low.shift(1)), 0)

    # Smooth the -DM with Wilder's smoothing method
    smooth_dm_minus = dm_minus.rolling(window=timeperiod).sum()

    # Calculate the True Range (TR)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

    # Smooth the TR with Wilder's smoothing method
    smooth_tr = tr.rolling(window=timeperiod).sum()

    # Avoid division by zero
    smooth_tr[smooth_tr == 0] = pd.NA

    # Calculate the Directional Indicator
    minus_di_ = 100 * (smooth_dm_minus / smooth_tr.fillna(method="ffill"))

    return minus_di_


def atr(high, low, close, timeperiod=14):
    """
    Calculate Average True Range (ATR).

    Parameters:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    close (pd.Series): Closing price series.
    timeperiod (int): Number of periods to consider for the calculation.

    Returns:
    pd.Series: Average True Range values.
    """
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (close.shift() - low).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR
    atr_ = tr.rolling(window=timeperiod).mean()

    return atr_


def MFI(high, low, close, volume, timeperiod=14):
    """
    Calculate Money Flow Index (MFI).

    Parameters:
    high (np.array): Array of high prices.
    low (np.array): Array of low prices.
    close (np.array): Array of closing prices.
    volume (np.array): Array of trading volumes.
    timeperiod (int): Number of periods to consider for the calculation.

    Returns:
    np.array: Array of Money Flow Index values.
    """
    # Calculate Typical Price
    typical_price = (high + low + close) / 3

    # Calculate Raw Money Flow
    raw_money_flow = typical_price * volume

    # Calculate Positive and Negative Money Flow
    positive_money_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_money_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

    # Calculate Money Ratio
    money_ratio = (
        positive_money_flow.rolling(window=timeperiod).sum() / negative_money_flow.rolling(window=timeperiod).sum()
    )

    # Calculate Money Flow Index
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def CCI(high, low, close, timeperiod=14):
    """
    Calculate Commodity Channel Index (CCI).

    Parameters:
    high (np.array): Array of high prices.
    low (np.array): Array of low prices.
    close (np.array): Array of closing prices.
    timeperiod (int): Number of periods to consider for the calculation.

    Returns:
    np.array: Array of Commodity Channel Index values.
    """
    # Typical Price
    typical_price = (high + low + close) / 3

    # Mean Deviation
    mean_typical_price = np.mean(typical_price, axis=0)
    mean_deviation = np.mean(np.abs(typical_price - mean_typical_price), axis=0)

    # Constant
    constant = 1 / (0.015 * timeperiod)

    # CCI Calculation
    cci = (typical_price - mean_typical_price) / (constant * mean_deviation)
    return cci


def LINEARREG_ANGLE(real, timeperiod=14):
    """
    Calculate the Linear Regression Angle for a given time period.

    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_LINEARREG_ANGLE.c

    :param real: NumPy ndarray of input data points.
    :param timeperiod: The number of periods to use for the regression (default is 14).
    :return: NumPy ndarray of angles in degrees.
    """
    # Validate input parameters
    if not isinstance(real, np.ndarray) or not isinstance(timeperiod, int):
        raise ValueError("Invalid input parameters.")
    if timeperiod < 2 or timeperiod > 100000:
        raise ValueError("timeperiod must be between 2 and 100000.")
    if len(real) < timeperiod:
        raise ValueError("Input data must have at least timeperiod elements.")

    # Initialize output array
    angles = np.zeros(len(real))

    # Calculate the total sum and sum of squares for the given time period
    SumX = timeperiod * (timeperiod - 1) * 0.5
    SumXSqr = timeperiod * (timeperiod - 1) * (2 * timeperiod - 1) / 6
    Divisor = SumX * SumX - timeperiod * SumXSqr

    # Calculate the angle for each point in the input array
    for today in range(timeperiod - 1, len(real)):
        SumXY = 0
        SumY = 0
        for i in range(timeperiod):
            SumY += real[today - i]
            SumXY += i * real[today - i]
        m = (timeperiod * SumXY - SumX * SumY) / Divisor
        angles[today] = np.arctan(m) * (180.0 / np.pi)

    return angles


def ACOS(real):
    """
        Calculate the vector trigonometric ACos of the input array.

        Parameters:
        real (np.ndarray): Input array of real values.

        Returns:
        np.ndarray: Array of calculated ACos values.
        """
    # Ensure the input is a numpy array
    real = np.asarray(real)

    # Calculate the ACos values
    acos_values = np.arccos(real)

    return acos_values


def ADD(real0, real1):
    """
    Calculate the vector arithmetic add of two input arrays.

    Parameters:
    real0 (np.ndarray): First input array of real values.
    real1 (np.ndarray): Second input array of real values.

    Returns:
    np.ndarray: Array of element-wise addition of real0 and real1.
    """
    # Ensure the inputs are numpy arrays
    real0 = np.asarray(real0)
    real1 = np.asarray(real1)

    # Check if the arrays have the same shape
    if real0.shape != real1.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Perform element-wise addition
    result = real0 + real1

    return result


import numpy as np


def AD(high, low, close, volume):
    """
    Calculate the Chaikin A/D Line.

    Parameters:
    high (np.ndarray): Array of high prices.
    low (np.ndarray): Array of low prices.
    close (np.ndarray): Array of closing prices.
    volume (np.ndarray): Array of trading volumes.

    Returns:
    np.ndarray: Array of Chaikin A/D Line values.
    """
    # Ensure the inputs are numpy arrays
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    volume = np.asarray(volume)

    # Check if all arrays have the same length
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("All input arrays must have the same length.")

    # Initialize variables
    ad = np.zeros_like(close)
    nbBar = len(close)
    outIdx = 0

    for currentBar in range(nbBar):
        high_val = high[currentBar]
        low_val = low[currentBar]
        tmp = high_val - low_val
        close_val = close[currentBar]

        if tmp > 0:
            ad[currentBar] = ad[currentBar - 1] + ((close_val - low_val - (high_val - close_val)) / tmp) * volume[
                currentBar]
        else:
            ad[currentBar] = ad[currentBar - 1]

    return ad


#TODO
def ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10):
    """
    Calculate the Chaikin A/D Oscillator.

    Parameters:
    high (np.ndarray): Array of high prices.
    low (np.ndarray): Array of low prices.
    close (np.ndarray): Array of closing prices.
    volume (np.ndarray): Array of trading volumes.
    fastperiod (int): Number of period for the fast MA (default 3).
    slowperiod (int): Number of period for the slow MA (default 10).

    Returns:
    np.ndarray: Array of Chaikin A/D Oscillator values.
    """
    # Ensure the inputs are numpy arrays
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    volume = np.asarray(volume)

    # Check if all arrays have the same length
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("All input arrays must have the same length.")

    # Initialize variables
    today = 0
    outIdx = 0
    lookbackTotal = 0
    slowestPeriod = max(fastperiod, slowperiod)
    ad = 0.0
    fastEMA = 0.0
    slowEMA = 0.0
    fastk = 2 / (fastperiod + 1)
    slowk = 2 / (slowperiod + 1)
    one_minus_fastk = 1 - fastk
    one_minus_slowk = 1 - slowk

    # Constants for EMA
    fastk = 2 / (fastperiod + 1)
    one_minus_fastk = 1 - fastk

    slowk = 2 / (slowperiod + 1)
    one_minus_slowk = 1 - slowk

    # Initialize the two EMA and skip the unstable period.
    while today < slowestPeriod:
        ad += ((close[today] - low[today]) - (high[today] - close[today])) / (high[today] - low[today]) * volume[today]
        fastEMA = (fastk * ad) + (one_minus_fastk * fastEMA)
        slowEMA = (slowk * ad) + (one_minus_slowk * slowEMA)
        today += 1

    # Perform the calculation for the requested range
    outReal = np.zeros_like(close)
    while today <= len(close) - 1:
        ad += ((close[today] - low[today]) - (high[today] - close[today])) / (high[today] - low[today]) * volume[today]
        fastEMA = (fastk * ad) + (one_minus_fastk * fastEMA)
        slowEMA = (slowk * ad) + (one_minus_slowk * slowEMA)
        outReal[outIdx] = fastEMA - slowEMA
        outIdx += 1
        today += 1

    return outReal


#TODO
def ADX(high, low, close, timeperiod=14):
    """
    Calculate the Average Directional Movement Index (ADX).

    Parameters:
    high (np.ndarray): Array of high prices.
    low (np.ndarray): Array of low prices.
    close (np.ndarray): Array of closing prices.
    timeperiod (int): Number of periods to use for calculation (default 14).

    Returns:
    np.ndarray: Array of ADX values.
    """
    # Ensure the inputs are numpy arrays
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    # Check if all arrays have the same length
    if not (len(high) == len(low) == len(close)):
        raise ValueError("All input arrays must have the same length.")

    # Initialize variables
    today = 0
    lookbackTotal = (2 * timeperiod) + 14 - 1  # Based on the C code's lookbackTotal calculation
    outReal = np.zeros_like(close)
    outIdx = 0

    # Initialize variables for calculation
    prevHigh = high[0]
    prevLow = low[0]
    prevClose = close[0]
    prevMinusDM = 0.0
    prevPlusDM = 0.0
    prevTR = 0.0

    # Calculate initial DM and TR
    for i in range(1, timeperiod):
        today += 1
        tempReal = high[today]
        diffP = tempReal - prevHigh
        prevHigh = tempReal

        tempReal = low[today]
        diffM = prevLow - tempReal
        prevLow = tempReal

        if diffM > 0 and diffP < diffM:
            prevMinusDM += diffM
        elif diffP > 0 and diffP > diffM:
            prevPlusDM += diffP

        true_range = max(tempReal - prevLow, prevHigh - tempReal, prevHigh - prevLow)
        prevTR += true_range
        prevClose = close[today]

    # Calculate the first ADX
    sumDX = 0.0
    for i in range(timeperiod):
        today += 1
        tempReal = high[today]
        diffP = tempReal - prevHigh
        prevHigh = tempReal

        tempReal = low[today]
        diffM = prevLow - tempReal
        prevLow = tempReal

        prevMinusDM -= prevMinusDM / timeperiod
        prevPlusDM -= prevPlusDM / timeperiod

        if diffM > 0 and diffP < diffM:
            prevMinusDM += diffM
        elif diffP > 0 and diffP > diffM:
            prevPlusDM += diffP

        true_range = max(tempReal - prevLow, prevHigh - tempReal, prevHigh - prevLow)
        prevTR = prevTR - (prevTR / timeperiod) + true_range
        prevClose = close[today]

        if prevTR != 0:
            minusDI = 100.0 * (prevMinusDM / prevTR)
            plusDI = 100.0 * (prevPlusDM / prevTR)
            sumDX += 100.0 * (abs(minusDI - plusDI) / (minusDI + plusDI))

    prevADX = sumDX / timeperiod

    # Output the first ADX
    outReal[outIdx] = prevADX
    outIdx += 1

    # Calculate and output subsequent ADX
    for today in range(timeperiod, len(close)):
        tempReal = high[today]
        diffP = tempReal - prevHigh
        prevHigh = tempReal

        tempReal = low[today]
        diffM = prevLow - tempReal
        prevLow = tempReal

        prevMinusDM -= prevMinusDM / timeperiod
        prevPlusDM -= prevPlusDM / timeperiod

        if diffM > 0 and diffP < diffM:
            prevMinusDM += diffM
        elif diffP > 0 and diffP > diffM:
            prevPlusDM += diffP

        true_range = max(tempReal - prevLow, prevHigh - tempReal, prevHigh - prevLow)
        prevTR = prevTR - (prevTR / timeperiod) + true_range
        prevClose = close[today]

        if prevTR != 0:
            minusDI = 100.0 * (prevMinusDM / prevTR)
            plusDI = 100.0 * (prevPlusDM / prevTR)
            tempReal = abs(minusDI - plusDI) / (minusDI + plusDI)
            prevADX = ((prevADX * (timeperiod - 1)) + tempReal) / timeperiod

        outReal[outIdx] = prevADX
        outIdx += 1

    return outReal[:outIdx]


def BBANDS(real, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0):
    """
    Calculate Bollinger Bands.

    Parameters:
    real (np.ndarray): Input array of real values.
    timeperiod (int): Number of periods for the moving average (default 5).
    nbdevup (float): The number of standard deviations to add to the moving average for the upper band (default 2.0).
    nbdevdn (float): The number of standard deviations to subtract from the moving average for the lower band (default 2.0).
    matype (int): Type of moving average (default 0 for Simple Moving Average).

    Returns:
    tuple: (upperband, middleband, lowerband) arrays.
    """
    # Ensure the input is a numpy array
    real = np.asarray(real)

    # Validate input parameters
    if timeperiod < 2 or timeperiod > 100000:
        raise ValueError("timeperiod must be between 2 and 100000.")
    if nbdevup < -3.000000e+37 or nbdevup > 3.000000e+37:
        raise ValueError("nbdevup must be between -3.000000e+37 and 3.000000e+37.")
    if nbdevdn < -3.000000e+37 or nbdevdn > 3.000000e+37:
        raise ValueError("nbdevdn must be between -3.000000e+37 and 3.000000e+37.")
    if matype < 0 or matype > 8:
        raise ValueError("matype must be between 0 and 8.")

    # Calculate the moving average (middle band)
    middleband = MA(real, timeperiod, matype)

    # Calculate the standard deviation
    temp = real[timeperiod-1:] - middleband[:-timeperiod+1]
    stddev = np.sqrt(np.mean(temp**2))

    # Calculate the upper and lower bands
    upperband = middleband + stddev * nbdevup
    lowerband = middleband - stddev * nbdevdn

    # Return the bands
    return upperband, middleband, lowerband

#TODO 计算结果不对
def MA(real, timeperiod=30, matype=0):
    """
    Calculate the moving average of the input data.

    Parameters:
    real (np.ndarray): Input array of real values.
    timeperiod (int): Number of periods for the moving average (default 30).
    matype (int): Type of moving average (default 0 for Simple Moving Average).

    Returns:
    np.ndarray: Array of moving average values.
    """
    # Ensure the input is a numpy array
    real = np.asarray(real)

    # Validate input parameters
    if timeperiod < 1 or timeperiod > 100000:
        raise ValueError("timeperiod must be between 1 and 100000.")
    if matype < 0 or matype > 8:
        raise ValueError("matype must be between 0 and 8.")

    # Handle the case where timeperiod is 1, in which case the moving average is just the input itself
    if timeperiod == 1:
        return real

    # Call the appropriate moving average function based on matype
    if matype == 0:  # Simple Moving Average (SMA)
        return np.convolve(real, np.ones(timeperiod), 'valid') / timeperiod
    # Add other moving average types here if needed

    # For simplicity, only SMA is implemented in this example.
    # You can extend this function to include other types of moving averages (EMA, WMA, etc.)
    # based on the TA-LIB documentation and the corresponding formulas.

    # Return the calculated moving average
    return np.convolve(real, np.ones(timeperiod), 'valid') / timeperiod


import numpy as np


#Momentum
#Rate of Change
def ROC(df, n):
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))
    df = df.join(ROC)
    return df

#Average True Range
def ATR(df, n):
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))
    df = df.join(ATR)
    return df

#Bollinger Bands
def BBANDS(df, n):
    MA = pd.Series(pd.rolling_mean(df['Close'], n))
    MSD = pd.Series(pd.rolling_std(df['Close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))
    df = df.join(B2)
    return df

#Pivot Points, Supports and Resistances
def PPSR(df):
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df

#Stochastic oscillator %K
def STOK(df):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    df = df.join(SOk)
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)
def STO(df,  nK, nD, nS=1):
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()
    df = df.join(SOk)
    df = df.join(SOd)
    return df
# Stochastic Oscillator, SMA smoothing, nS = slowing (1 if no slowing)
def STO(df, nK, nD,  nS=1):
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))
    SOd = pd.Series(SOk.rolling(window=nD, center=False).mean(), name = 'SO%d'+str(nD))
    SOk = SOk.rolling(window=nS, center=False).mean()
    SOd = SOd.rolling(window=nS, center=False).mean()
    df = df.join(SOk)
    df = df.join(SOd)
    return df
#Trix
def TRIX(df, n):
    EX1 = pd.ewma(df['Close'], span = n, min_periods = n - 1)
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))
    df = df.join(Trix)
    return df

#Average Directional Movement Index
def ADX(df, n, n_ADX):
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df

#MACD, MACD Signal and MACD difference
def MACD(df, n_fast, n_slow):
    EMAfast = pd.Series(pd.ewma(df['Close'], span = n_fast, min_periods = n_slow - 1))
    EMAslow = pd.Series(pd.ewma(df['Close'], span = n_slow, min_periods = n_slow - 1))
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(pd.ewma(MACD, span = 9, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df

#Mass Index
def MassI(df):
    Range = df['High'] - df['Low']
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)
    Mass = EX1 / EX2
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')
    df = df.join(MassI)
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def Vortex(df, n):
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))
    df = df.join(VI)
    return df





#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df

#Relative Strength Index
def RSI(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
    df = df.join(RSI)
    return df

#True Strength Index
def TSI(df, r, s):
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))
    df = df.join(TSI)
    return df

#Accumulation/Distribution
def ACCDIST(df, n):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    df = df.join(AD)
    return df

#Chaikin Oscillator
def Chaikin(df):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')
    df = df.join(Chaikin)
    return df

#Money Flow Index and Ratio
def MFI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))
    df = df.join(MFI)
    return df

#On-balance Volume
def OBV(df, n):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:
            OBV.append(df.get_value(i + 1, 'Volume'))
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:
            OBV.append(0)
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:
            OBV.append(-df.get_value(i + 1, 'Volume'))
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))
    df = df.join(OBV_ma)
    return df

#Force Index
def FORCE(df, n):
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))
    df = df.join(F)
    return df

#Ease of Movement
def EOM(df, n):
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))
    df = df.join(Eom_ma)
    return df

#Commodity Channel Index
def CCI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))
    df = df.join(CCI)
    return df

#Coppock Curve
def COPP(df, n):
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))
    df = df.join(Copp)
    return df

#Keltner Channel
def KELCH(df, n):
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df

#Ultimate Oscillator
def ULTOSC(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')
    df = df.join(UltO)
    return df

#Donchian Channel
def DONCH(df, n):
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))
    DonCh = DonCh.shift(n - 1)
    df = df.join(DonCh)
    return df

#Standard Deviation
def STDDEV(df, n):
    df = df.join(pd.Series(pd.rolling_std(df['Close'], n), name = 'STD_' + str(n)))
    return df


def SAR(high, low, acceleration=0.02, maximum=0.2):
    sar = np.zeros(len(high))
    is_long = None
    ep = 0
    af = acceleration
    sar[0] = low[0]  # 初始化第一个SAR值

    for i in range(1, len(high)):
        new_high = high[i]
        new_low = low[i]
        prev_high = high[i-1]
        prev_low = low[i-1]

        if i == 1:
            if high[i] > high[i-1]:
                is_long = True
                ep = high[i-1]
                sar[i] = low[i]
            else:
                is_long = False
                ep = low[i-1]
                sar[i] = high[i]

        else:
            if is_long:
                if new_low <= sar[i-1]:
                    is_long = False
                    sar[i] = ep
                    ep = new_low
                    af = acceleration
                else:
                    ep = max(ep, new_high)
                    sar[i] = sar[i-1] + af * (ep - sar[i-1])
                    af = min(af + acceleration, maximum)

                if sar[i] < min(prev_low, new_low):
                    sar[i] = min(prev_low, new_low)
            else:
                if new_high >= sar[i-1]:
                    is_long = True
                    sar[i] = ep
                    ep = new_high
                    af = acceleration
                else:
                    ep = min(ep, new_low)
                    sar[i] = sar[i-1] + af * (ep - sar[i-1])
                    af = min(af + acceleration, maximum)

                if sar[i] > max(prev_high, new_high):
                    sar[i] = max(prev_high, new_high)

    return sar

# 示例数据
# high = np.array([100, 102, 101, 105, 107, 106])
# low = np.array([99, 98, 99, 103, 104, 105])
#
# # 计算SAR
# sar_values = SAR(high, low)
# print(sar_values)



def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):

    # 计算%K (FastK)
    length = len(high)

    # 计算%K (FastK)
    lowest_low = np.min(low[max(0, length - fastk_period):], axis=0)
    highest_high = np.max(high[max(0, length - fastk_period):], axis=0)
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k[fastk_period - 1:] = np.nan  # 将计算%K所需的初始周期部分设置为NaN

    # 计算%D (SlowD)
    if slowk_matype == 0:  # 简单移动平均
        slowk = np.convolve(k, np.ones(slowk_period) / slowk_period, mode='valid')
    else:
        # 其他类型的移动平均可以根据需要添加
        raise ValueError("Only SMA is supported for slowk_matype")

    if slowd_matype == 0:  # 简单移动平均
        slowd = np.convolve(slowk, np.ones(slowd_period) / slowd_period, mode='valid')
    else:
        # 其他类型的移动平均可以根据需要添加
        raise ValueError("Only SMA is supported for slowd_matype")

    return slowk, slowd



