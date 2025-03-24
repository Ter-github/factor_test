import pandas as pd
import numpy as np

# 定义函数：获取前一个交易日
def get_previous_trading_date(current_date, trading_dates):
    idx = trading_dates.index(current_date)
    return trading_dates[idx - 1] if idx > 0 else None

def process_day(group, symbol,trading_dates):
    """
    按照日盘和夜盘时间范围划分交易数据，并对每段数据进行前后切片处理，最后返回日盘和夜盘的 baskets 和 window_size。

    :param group: 分组后的 DataFrame，每组是一个交易日的数据。
    :param prev_period: 去掉每段数据前 prev_period 条记录。
    :param back_period: 去掉每段数据后 back_period 条记录。
    :param trading_dates: 所有交易日的序列，用于查找前一个交易日。
    :param V: 每个桶的目标交易量（可以根据需要调整）
    :return: 处理后的日盘和夜盘数据分别处理后的 DataFrame。
    """
    # 获取当前交易日和前一个交易日
    trading_date = group['trading_date'].iloc[0]
    previous_trading_date = get_previous_trading_date(trading_date, trading_dates)

    # 定义时间范围
    if symbol == 'ag':
        day_start = pd.to_datetime(f"{trading_date} 09:00:00")
        day_end = pd.to_datetime(f"{trading_date} 14:57:00")
        night_start = pd.to_datetime(f"{previous_trading_date} 21:00:00") if previous_trading_date else None
        night_end = (pd.to_datetime(f"{previous_trading_date} 02:27:00") + pd.Timedelta(days=1)) if previous_trading_date else None
    else:
        day_start = pd.to_datetime(f"{trading_date} 09:00:00")
        day_end = pd.to_datetime(f"{trading_date} 15:00:00")
        night_start = pd.to_datetime(f"{previous_trading_date} 21:00:00") if previous_trading_date else None
        night_end = (pd.to_datetime(f"{previous_trading_date} 22:57:00")) if previous_trading_date else None

    # 筛选日盘数据
    day_session = group[(group['exchange_time'] >= day_start) & (group['exchange_time'] <= day_end)]

    # 筛选夜盘数据（需要判断是否有前一个交易日）
    if night_start:
        night_session = group[(group['exchange_time'] >= night_start) & (group['exchange_time'] <= night_end)]
    else:
        night_session = pd.DataFrame()  # 如果没有前一个交易日，则夜盘数据为空

    # day_session_processed = day_session.iloc[prev_period:-back_period] 

    # # 如果夜盘数据存在，则进行处理；否则跳过
    # if not night_session.empty:
    #     night_session_processed = night_session.iloc[prev_period:-back_period]
    # else:
    #     night_session_processed = pd.DataFrame()  # 为空时可以直接跳过处理

    # # 在日盘和夜盘数据上分别应用桶划分逻辑
    # def bucketize_data(session_data):
    #     # 如果 session_data 不为空并且包含 'Volume' 列
    #     if session_data.empty or 'Volume' not in session_data.columns:
    #         return session_data  # 返回原始数据，因为数据为空或者没有 'Volume' 列

    #     # session_data['current_volume'] = session_data['Volume'].diff()
    #     # session_data['current_volume'].fillna(0, inplace=True)
    #     current_basket = 0  # 当前桶的交易量
    #     window_size = 0  # 当前桶的起始索引

    #     current_basket_list = []
    #     window_size_list = []


    #     # 遍历 `current_volume` 数据，将数据划分为多个桶
    #     for volume in session_data['current_volume'].values:
    #         current_basket += volume  # 累积当前桶的交易量
    #         # current_basket_list.append(current_basket)
    #         window_size += 1  # 增加窗口大小
    #         # window_size_list.append(window_size)

    #         # 当当前桶的交易量达到或超过目标交易量时
    #         if current_basket >= V:
    #             current_basket_list.extend([current_basket]*window_size)
    #             window_size_list.extend([window_size]*window_size)
    #             window_size = 0
    #             current_basket = 0

    #     current_basket_list.extend([current_basket]*window_size)
    #     window_size_list.extend([window_size]*window_size)

    #     session_data['basket_volume'] = current_basket_list
    #     session_data['window_size'] = window_size_list


    #     return session_data

    # # 分别对日盘和夜盘数据进行桶划分处理
    # day_session_processed = bucketize_data(day_session_processed)

    # if not night_session_processed.empty:
    #     night_session_processed = bucketize_data(night_session_processed)

    # 拼接处理后的日盘和夜盘数据
    # processed_data = pd.concat([night_session,day_session], ignore_index=True)

    # 返回处理后的日盘和夜盘数据
    return night_session,day_session