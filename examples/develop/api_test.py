import czsc
from loguru import logger
# 默认会输出 log，可以用下面这行关闭 log 输出
logger.disable('czsc')
# 首次使用需要设置对应 url 的 token
# czsc.set_url_token(token="qCfHxwHchMb8pHif3jYy3rrAQhdXzhTA", url='http://api.waditu.com')
# 初始化 tushare 数据客户端，并设置缓存路径
# tdc = czsc.DataClient(url="http://api.waditu.com", cache_path=".quant_data_cache_waditu")
# tushare 数据接口定义参考：https://tushare.pro/document/2
# data = tdc.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# 这里的 token 可以联系韩知辰获取
czsc.set_url_token(token="qCfHxwHchMb8pHif3jYy3rrAQhdXzhTA", url="http://zbczsc.com:9106")
# 初始化自建的数据客户端，并设置缓存路径
dc = czsc.DataClient(url="http://zbczsc.com:9106", cache_path=".quant_data_cache_zbczsc")
# 自定义数据API参考子页面目录
data = dc.stock_basic(status=1, fields='code,name,sdt,edt,status')
# data = zdc.etf_basic(v=2, fields='code,name')
# data = dc.index_dailybasic(code="600000.SH", sdt='20100101', edt='20101231')
# data = dc.cashflow(code="600001.SH")

# data = dc.daily_basic(trade_date="20140411")
print(data.head())