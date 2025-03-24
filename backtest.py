import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from factor_install import *
from process_day import *

def backtest_value(day_df,night_df,symbol,threshold,current_vol_threshold,factor_name):
    long_threshold = threshold
    short_threshold = -threshold

    position_limit = 150

    tax_dict = {
        'tax_ag':0.00005,
        'tax_sp':0,
        'tax_rb':0.0001,
        'tax_ru':0.0002,
        'tax_hc':0.0001,
        'tax_fu':0.00005,
        'tax_cu':0.0001
        }
    
    tax = tax_dict[f"tax_{symbol}"]

    day_pnl, night_pnl = 0, 0
    day_position,night_position = 0,0

    day_df.reset_index(inplace=True)
    night_df.reset_index(inplace=True)
    
    day_df['signal'] = ((day_df[factor_name] >= long_threshold) & (day_df['log_current_volume'] >= current_vol_threshold))*1 - ((day_df[factor_name] <= short_threshold) & (day_df['log_current_volume'] >= current_vol_threshold))*1
    
    for i in range(len(day_df)):
        if (day_position >= position_limit) and (day_df.at[i,'signal'] > 0):
            day_df.at[i,'signal'] = 0
        elif (day_position <= -position_limit) and (day_df.at[i,'signal'] < 0):
            day_df.at[i,'signal'] = 0
        elif (day_position <= position_limit) and (day_position >= -position_limit):
            day_position += day_df.at[i,'signal']

    # day_last_position = day_df['signal'].values[-1]
    day_last_position = day_position
    day_pnl += (-(day_df['AskPrice1'] * (day_df['signal']==1)).sum() + (day_df['BidPrice1'] * (day_df['signal']==-1)).sum() - (day_df['AskPrice1'] * (day_df['signal']==1) * tax).sum() - (day_df['BidPrice1'] * (day_df['signal']==-1) * tax).sum())
    day_force_price = 'AskPrice1' if day_last_position < 0 else 'BidPrice1'
    day_pnl += day_df[day_force_price].values[-1] * day_last_position

            



    

    if not night_df.empty:
        night_df['signal'] = ((night_df[factor_name] >= long_threshold) & (night_df['log_current_volume'] >= current_vol_threshold))*1 - ((night_df[factor_name] <= short_threshold) & (night_df['log_current_volume'] >= current_vol_threshold))*1
        
        for i in range(len(night_df)):
            if (night_position >= position_limit) and (night_df.at[i,'signal'] > 0):
                night_df.at[i,'signal'] = 0
            elif (night_position <= -position_limit) and (night_df.at[i,'signal'] < 0):
                night_df.at[i,'signal'] = 0
            elif (night_position <= position_limit) and (night_position >= -position_limit):
                night_position += night_df.at[i,'signal']

        # night_last_position = night_df['signal'].values[-1]
        night_last_position = night_position
        night_pnl += (-(night_df['AskPrice1'] * (night_df['signal']==1)).sum() + (night_df['BidPrice1'] * (night_df['signal']==-1)).sum() - (night_df['AskPrice1'] * (night_df['signal']==1) * tax).sum() - (night_df['BidPrice1'] * (night_df['signal']==-1) * tax).sum())
        night_force_price = 'AskPrice1' if night_last_position < 0 else 'BidPrice1'
        night_pnl += night_df[night_force_price].values[-1] * night_last_position



    # day_df['signal'] = ((day_df[factor_name] >= long_threshold) & (day_df['log_current_volume'] >= current_vol_threshold))*1 - ((day_df[factor_name] <= short_threshold) & (day_df['log_current_volume'] >= current_vol_threshold))*1
    # day_df['position'] = day_df['signal'].where(day_df['signal']!= 0, np.nan)
    # day_df['position'].fillna(method='ffill',inplace=True)
    # day_df['position_change'] = day_df['position'].diff()
    # day_df['position_change'].fillna(0,inplace=True)
    # day_last_position = day_df['position'].values[-1]

    # day_pnl += (-(day_df[day_df['position_change'] > 0]['AskPrice1'] * day_df[day_df['position_change'] > 0]['position_change']).sum() - (day_df[day_df['position_change'] < 0]['BidPrice1'] * day_df[day_df['position_change'] < 0]['position_change']).sum() - ((day_df[day_df['position_change'] > 0]['AskPrice1'] * day_df[day_df['position_change'] > 0]['position_change']) * tax).sum() + ((day_df[day_df['position_change'] < 0]['BidPrice1'] * day_df[day_df['position_change'] < 0]['position_change']) * tax).sum())
    # day_force_price = 'AskPrice1' if day_last_position < 0 else 'BidPrice1'
    # day_pnl += day_df[day_force_price].values[-1] * day_last_position
    

    # if not night_df.empty:
    #     night_df['signal'] = ((night_df[factor_name] >= long_threshold) & (night_df['log_current_volume'] >= current_vol_threshold))*1 - ((night_df[factor_name] <= short_threshold) & (night_df['log_current_volume'] >= current_vol_threshold))*1
    #     night_df['position'] = night_df['signal'].where(night_df['signal']!= 0, np.nan)
    #     night_df['position'].fillna(method='ffill',inplace=True)
    #     night_df['position_change'] = night_df['position'].diff()
    #     night_df['position_change'].fillna(0,inplace=True)
    #     night_last_position = night_df['position'].values[-1]

    #     night_pnl += (-(night_df[night_df['position_change'] > 0]['AskPrice1'] * night_df[night_df['position_change'] > 0]['position_change']).sum() - (night_df[night_df['position_change'] < 0]['BidPrice1'] * night_df[night_df['position_change'] < 0]['position_change']).sum() - ((night_df[night_df['position_change'] > 0]['AskPrice1'] * night_df[night_df['position_change'] > 0]['position_change']) * tax).sum() + ((night_df[night_df['position_change'] < 0]['BidPrice1'] * night_df[night_df['position_change'] < 0]['position_change']) * tax).sum())
    #     night_force_price = 'AskPrice1' if night_last_position < 0 else 'BidPrice1'
    #     night_pnl += night_df[night_force_price].values[-1] * night_last_position

    # sum_pnl = day_pnl + night_pnl
    # high = day_df['AskPrice1'].max()


    # with open('test.csv','a+') as f:
    #     f.writelines(f'{sum_pnl/high}\n')

    sum_pnl = day_pnl + night_pnl


    with open('test.csv','a+') as f:
        f.writelines(f'{sum_pnl}\n')
    
    if not night_df.empty:
        b_value = sum_pnl/(day_df['signal'].abs().sum() + night_df['signal'].abs().sum())
    else:
        b_value = sum_pnl/(day_df['signal'].abs().sum())

    return b_value



def backtest(df,symbol,start_time,end_time,threshold,current_vol_threshold,factor_name):
    test_start_time = pd.to_datetime(start_time)
    test_end_time = pd.to_datetime(end_time)

    test_table = df[(df['trading_date'] >= test_start_time) & (df['trading_date'] <= test_end_time)]

    # 当 AskPrice1 为 0 时，用 BidPrice1 替换
    test_table['AskPrice1'] = test_table['AskPrice1'].where(test_table['AskPrice1'] != 0, test_table['BidPrice1'])

    # 当 AskPrice1 为 0 时，用 AskPrice1 替换
    test_table['BidPrice1'] = test_table['BidPrice1'].where(test_table['BidPrice1'] != 0, test_table['AskPrice1'])

    # 计算一些基本信息
    test_table['mid_price'] = (test_table['BidPrice1'] + test_table['AskPrice1']) / 2
    test_table['current_volume'] = test_table['Volume'].diff()

    with open('test.csv','w+') as f:
        f.writelines(f'day_ret\n')
    # 按 'trading_date' 分组，使用 process_day 处理每个分组
    new_test_table = test_table.copy()
    unique_trading_dates = sorted(new_test_table['trading_date'].unique())
    result = new_test_table.groupby('trading_date').apply(process_day,symbol = symbol,trading_dates=unique_trading_dates)
    # 处理结果：返回每一天的日盘和夜盘数据以及合并后的结果
    # concat_results = []
    backtest_value_list = []

    # 处理每一天的日盘和夜盘数据
    for night_df,day_df in result:
        day_df['frt_120'] = -day_df['mid_price'].diff(-120)
        day_df['frt_120'].fillna(0,inplace=True)
        factor_install(day_df,symbol)
        if not night_df.empty:
            night_df['frt_120'] = -night_df['mid_price'].diff(-120)
            night_df['frt_120'].fillna(0,inplace=True)
            factor_install(night_df,symbol)

        backtest = backtest_value(day_df,night_df,symbol = symbol,threshold=threshold,current_vol_threshold = current_vol_threshold,factor_name=factor_name) 
        backtest_value_list.append(backtest)

        # concat_df = pd.concat([night_df,day_df],ignore_index=True)
        # concat_results.append(concat_df)

    # test_data = pd.concat(concat_results, ignore_index=True)

    df1 = pd.read_csv('test.csv')
    df1['value'] = df1['day_ret'].cumsum()
    df1['cumulative_max'] = df1['value'].cummax()
    df1['drawdown'] = (df1['value'] - df1['cumulative_max'])
    # 计算最大回撤
    max_drawdown = df1['drawdown'].min()
    print(f"最大回撤: {max_drawdown:.2f}")
    df1['value'].plot()
    # 添加标题和轴标签
    plt.title('累计收益曲线')
    plt.xlabel('时间 t')
    plt.ylabel('累计收益')

    # 显示图表
    plt.show()

    return np.array(backtest_value_list).mean()