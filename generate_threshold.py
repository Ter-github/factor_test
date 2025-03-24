import pandas as pd
import numpy as np
from factor_install import *
from process_day import *

def generate_threshold(df,symbol,start_time,end_time,factor_name):
    # 设置开始和结束时间
    train_start_time = pd.to_datetime(start_time)
    train_end_time = pd.to_datetime(end_time)

    table = df[(df['trading_date'] >= train_start_time) & (df['trading_date'] <= train_end_time)]

    # 当 AskPrice1 为 0 时，用 BidPrice1 替换
    table['AskPrice1'] = table['AskPrice1'].where(table['AskPrice1'] != 0, table['BidPrice1'])

    # 当 AskPrice1 为 0 时，用 AskPrice1 替换
    table['BidPrice1'] = table['BidPrice1'].where(table['BidPrice1'] != 0, table['AskPrice1'])

    # 计算一些基本信息
    table['mid_price'] = (table['BidPrice1'] + table['AskPrice1']) / 2
    table['current_volume'] = table['Volume'].diff()

    new_table = table.copy()
    unique_trading_dates = sorted(new_table['trading_date'].unique())
    result = new_table.groupby('trading_date').apply(process_day,symbol = symbol,trading_dates=unique_trading_dates)
    # 处理结果：返回每一天的日盘和夜盘数据以及合并后的结果
    concat_results = []



    # 处理每一天的日盘和夜盘数据
    for night_df,day_df in result:
        day_df['frt_120'] = -day_df['mid_price'].diff(-120)
        day_df['frt_120'].fillna(0,inplace=True)
        factor_install(day_df,symbol)

        if not night_df.empty:
            night_df['frt_120'] = -night_df['mid_price'].diff(-120)
            night_df['frt_120'].fillna(0,inplace=True)
            factor_install(night_df,symbol)

        concat_df = pd.concat([night_df,day_df],ignore_index=True)
        # new_mean = concat_df['factor'].mean()
        # new_std = concat_df['factor'].std()
        # day_results.append(day_df)
        # night_results.append(night_df)
        concat_results.append(concat_df)


    # 合并所有日盘和夜盘数据
    # final_day_data = pd.concat(day_results, ignore_index=True)
    # final_night_data = pd.concat(night_results, ignore_index=True)
    train_data = pd.concat(concat_results, ignore_index=True)
    print(train_data[factor_name].quantile([0.01,0.99]))
    print(f'成交量阈值为：', train_data['log_current_volume'].quantile(0.6))
    threshold = (train_data[factor_name].quantile(0.99) - train_data[factor_name].quantile(0.01))/2
    current_vol_threshold = train_data['log_current_volume'].quantile(0.6)
    return threshold,current_vol_threshold