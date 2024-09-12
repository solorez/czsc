import pandas as pd
from czsc import CzscStrategyBase , CzscTrader
from czsc.utils import get_sub_elements, create_single_signal
from czsc.analyze import CZSC
from czsc.signals.tas import update_ma_cache
from collections import OrderedDict


def ma_base_V240601(c: CZSC, **kwargs) -> OrderedDict:
    """MA 多空和方向信号

    参数模板："{freq}_D{di}{ma_type}#{timeperiod}_分类V240601"

    **信号逻辑：**

    1. close > ma，多头；反之，空头

    **信号列表：**

    - Signal('15分钟_D1SMA#5_分类V240601_空头_任意_任意_0')
    - Signal('15分钟_D1SMA#5_分类V240601_多头_任意_任意_0')

    :param c: CZSC对象
    :param kwargs:
        - di: 信号计算截止倒数第i根K线
        - ma_type: 均线类型，必须是 `ma_type_map` 中的 key
        - timeperiod: 均线计算周期
    :return:
    """
    di = int(kwargs.get("di", 1))
    ma_type = kwargs.get("ma_type", "SMA").upper()
    timeperiod = int(kwargs.get("timeperiod", 5))
    freq = c.freq.value
    k1, k2, k3 = f"{freq}_D{di}{ma_type}#{timeperiod}_分类V240601".split("_")

    key = update_ma_cache(c, ma_type=ma_type, timeperiod=timeperiod)
    bars = get_sub_elements(c.bars_raw, di=di, n=3)
    v1 = "多头" if bars[-1].close >= bars[-1].cache[key] else "空头"
    # v2 = "向上" if bars[-1].cache[key] >= bars[-2].cache[key] else "向下"
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)   #, v2=v2

