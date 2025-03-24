import pandas as pd
import numpy as np
from scipy import fft


def create_lagged_features(df, window , factor):
    new_factor_list = []
    for i in window:
        for j in factor:
            df[f'{j}_lag{i}'] = df[j].rolling(i).mean()
            df[f'{j}_lag{i}'].fillna(0,inplace = True)
            new_factor_list.append(f'{j}_lag{i}')
    return df,new_factor_list

def factor_install(table,symbol):
    table['BAV_diff'] = table['BidVolume1'].diff() - table['AskVolume1'].diff()
    table['BAV_diff_transform'] = (
    1
    + 0.2 * (np.abs(table['BAV_diff']) >= 6*15)
    * np.log(np.abs(table['BAV_diff'] / (6*15)))
    ) * (1 / (1 + np.exp(-table['BAV_diff'] / 15)) - 0.5)
    table['BAV_diff_transform'].fillna(0,inplace=True)

    table['delta_vol_Bid'] = table['BidVolume1'].diff() * (table['BidPrice1'] == table['BidPrice1'].shift(1)) + table['BidVolume1'] *(table['BidPrice1'] > table['BidPrice1'].shift(1)) + (table['BidVolume1']-table['BidVolume2'].shift(1)) * (table['BidPrice1'] < table['BidPrice1'].shift(1))
    table['delta_vol_Ask'] = table['AskVolume1'].diff() * (table['AskPrice1'] == table['AskPrice1'].shift(1)) + table['AskVolume1'] *(table['AskPrice1'] < table['AskPrice1'].shift(1)) + (table['AskVolume1']-table['AskVolume2'].shift(1)) * (table['AskPrice1'] > table['AskPrice1'].shift(1))
    table['Volume_Order_Imbalance'] = table['delta_vol_Bid'] - table['delta_vol_Ask']
    table['Volume_Order_Imbalance'].fillna(0,inplace=True)

    table['Base_factor'] = -(table['AskVolume1']-table['BidVolume1'])/(table['AskVolume1']+table['BidVolume1'])

    table['ratio'] = table['BidVolume1']/(table['BidVolume1'] + table['AskVolume1'])
    table['pending_vol_ratio_factor'] = 0.1 * table['BidPrice1'].diff(2)/(table['BidPrice1'] - table['BidPrice2']) + table['ratio'].diff(2)
    table['pending_vol_ratio_factor'].fillna(0,inplace=True)

    table['BidVolume'] = table[['BidVolume1','BidVolume2','BidVolume3','BidVolume4','BidVolume5']].sum(axis=1)
    table['AskVolume'] = table[['AskVolume1','AskVolume2','AskVolume3','AskVolume4','AskVolume5']].sum(axis=1)
    table['Bid_ratio'] = table['BidVolume']/table['BidVolume1']
    table['Ask_ratio'] = table['AskVolume']/table['AskVolume1']
    table['relative_vol_ratio_diff'] = table['Ask_ratio'] - table['Bid_ratio']
    table['relative_vol_ratio_imbalance'] = (table['relative_vol_ratio_diff'] - table['relative_vol_ratio_diff'].rolling(20,min_periods=1).mean())/table['relative_vol_ratio_diff'].rolling(20,min_periods=1).std()
    table['relative_vol_ratio_imbalance'].replace([np.inf, -np.inf], np.nan, inplace=True)
    table['relative_vol_ratio_imbalance'].fillna(0,inplace=True)

    table['Bid_submit_price'] = (table['BidPrice1'] * table['BidVolume1'] + table['BidPrice2'] * table['BidVolume2'] + table['BidPrice3'] * table['BidVolume3'] + table['BidPrice4'] * table['BidVolume4'] + table['BidPrice5'] * table['BidVolume5'])/table['BidVolume']
    table['Ask_submit_price'] = (table['AskPrice1'] * table['AskVolume1'] + table['AskPrice2'] * table['AskVolume2'] + table['AskPrice3'] * table['AskVolume3'] + table['AskPrice4'] * table['AskVolume4'] + table['AskPrice5'] * table['AskVolume5'])/table['AskVolume']
    table['std_factor'] = (table['Ask_submit_price'] - table['mid_price']) - (table['mid_price'] - table['Bid_submit_price'])
    table['submit_price_imbalance'] = (table['std_factor'] - table['std_factor'].rolling(10,min_periods=1).mean())/table['std_factor'].rolling(10,min_periods=1).std()
    table['submit_price_imbalance'].replace([np.inf, -np.inf], np.nan, inplace=True)
    table['submit_price_imbalance'].fillna(0, inplace=True)

    table['frt_120'] = -table['mid_price'].diff(-120)


    table['log_current_volume'] = np.log(table['current_volume']+1)




    # # 提取因子和目标变量
    # factor_columns = ['Base_factor','BAV_diff_transform', 'Volume_Order_Imbalance',
    #                   'pending_vol_ratio_factor', 'submit_price_imbalance', 'relative_vol_ratio_imbalance']

    # 提取因子和目标变量
    factor_columns = ['Base_factor','BAV_diff_transform', 'Volume_Order_Imbalance','pending_vol_ratio_factor', 'relative_vol_ratio_imbalance']

    # table,lag_factor = create_lagged_features(table,window = [10,], factor= factor_columns)
    # factor_columns = lag_factor

    # table,lag_factor = create_lagged_features(table,window = [10,26,40,60,120], factor=factor_columns)
    # factor_columns = lag_factor

    mask = (table[['AskPrice1','AskPrice2','AskPrice3','AskPrice4','AskPrice5','BidPrice1','BidPrice2','BidPrice3','BidPrice4','BidPrice5']]==0).any(axis=1)
    table['log_current_volume'] = table['log_current_volume'].where(~mask,0)
    table[factor_columns] = table[factor_columns].where(~mask,0)
    table[factor_columns].fillna(0,inplace=True)



    # 五因子模型（不加窗口期）
    weights_dict = {
        'weights_ag_with_base':[0.2459649489598114, 0.16081059208038484, 0.0028959203634215296, 0.07037271178061585, 0.03870002475424947],
        'weights_sp_with_base':[0.5400959637227973, 0.3376072745098696, 0.0024819388828543295, 0.2065826480612081, 0.09555345245479584],
        'weights_rb_with_base':[0.32950569541043406, 0.07473829156901622, 0.0005120645718051671, 0.007495562938061961, 0.04283780673118161],
        'weights_ru_with_base':[1.3660804704769458, 1.1975637932322514, 0.0077960986303946345, 0.3756347394385109, 0.24541049749830957],
        'weights_hc_with_base':[0.3397201050510739, 0.07965616521323109, 0.0008482954528104843, -0.039778787956565004, 0.039948264412887534],
        'weights_fu_with_base':[0.3176169480052628, 0.16075411107861498, 0.0010171414193556113, 0.08695752511301705, 0.034569678076322406],
        'weights_cu_with_base':[2.6520552764798824, 1.8444801731519749, 0.05394403419292516, 0.1396110776862445, 0.4007462031554879]
        }

    # 五因子模型（单窗口期：[10,]）
    # weights_dict = {
    #     'weights_ag_with_base':[0.1667287657251836, 1.3609398279149973, 0.0011721621496397383, 0.8631162331726975, 0.014854453744628708],
    #     'weights_sp_with_base':[0.4190766823288031, 2.9166041117431782, -0.004754504338588828, 3.2195925979878086, 0.0234058616366879],
    #     'weights_rb_with_base':[0.23327073512682828, 0.6009786133673489, -0.00016644581085667476, 1.7935429623386243, 0.014706154006853805],
    #     'weights_ru_with_base':[1.115447959320505, 9.460848168723894, -0.014446352533269015, 7.630875120111201, 0.06270628156294521],
    #     'weights_hc_with_base':[0.25460389562441876, 0.9852656745679826, -0.0007161654431703774, 1.4566264912518003, 0.006318077093739149],
    #     'weights_fu_with_base':[0.2447690440146759, 1.1700526587455888, -0.0015991526842266681, 1.792758643115566, -0.00430222507218492],
    #     'weights_cu_with_base':[1.6876079018067431, 34.725403170495845, 0.0466512021191831, 1.4976080608055327, 0.13926651298799103]
    #     }

    # 五因子模型（多窗口期：[10,26,40,60,120]）
    # weights_dict = {
    #     'weights_ag_with_base':[0.03778280326446649, 1.2995969281733417, 0.000835613624328196, 0.6423281612975371, 0.02129462625880559, 0.06437354366762715, 0.020775541454085, -0.0007953840220009909, 1.665198552236512, 0.02253718227306704, 0.17940567335519236, -0.1955667131365502, 0.0007849295196266536, 0.8234286712845899, -0.020822237982379883, -0.07162298644513956, 0.634575068723883, -0.0009429613424911, -1.6873785250720084, 0.013627827825085889, -0.06347990844525091, 0.04686438604027649, 0.0006800403195371904, 1.5783799413669684, -0.04016378672283313],
    #     'weights_sp_with_base':[0.0502348031039906, 2.542383317678296, -0.004603592036642324, 2.050233116483331, 0.035966973050544596, 0.0660651695790319, 0.2980976835459847, -9.370887898382734e-05, 4.806334109428076, 0.009966190511480736, 0.295111315414846, 0.23391591946099294, -0.002393004861058159, 4.441726794854888, 0.019138860967112103, 0.1620394763186497, 0.2871809520067634, -0.004403097122234131, 1.5427532597481552, 0.001270865581268342, -0.2666608984716827, 1.4544524085994794, 0.0015407413072419263, 0.7352706562183867, -0.2339977523010354],
    #     'weights_rb_with_base':[-0.04031733348380519, 0.5019670060582897, 6.371215726826657e-05, 0.8506656485051094, 0.011140498307920334, -0.03518264484478983, 0.17148612535425312, -0.0002050378861700265, 2.5577587773533845, 0.01477345667731837, 0.12399873704670548, 0.002302567859478055, -0.00010658549507999083, 3.2910901403373516, 0.024810364452683857, 0.20003663402883376, 0.06860974761579469, -0.0006565438870397725, 3.102998445832379, 0.00810833187242534, -0.07227773639307761, 0.009419570509479811, -0.002686884337453419, 4.399085060823657, 0.0017643763863735167],
    #     'weights_ru_with_base':[-0.3713497468968509, 7.329024693783999, -0.011683562513829309, 1.882487366354332, 0.09626337332984042, -0.1335012176810907, 2.031752712146018, -0.0017769000102268644, 12.266638479890098, -0.005063387092245411, 0.4660597478051512, 0.7132577102871981, -0.005587324499444538, 18.202495574393513, 0.09823790084270748, 0.8144870963779686, 2.368357806569401, -0.020767099812588365, 18.33105083814785, -0.011351455982933715, 0.3762500379532502, 7.051190195067613, -0.04664510650308634, 32.79310085517755, -0.5025400994514579],
    #     'weights_hc_with_base':[0.004668754179483208, 0.7340448706361025, -0.00026783817603598484, 0.7032877139472542, 0.01185552624207904, -0.010366741415126408, 0.30115523694552654, -0.0007355889753171851, 2.2707558528008223, 0.002761099098373209, 0.10015244584944458, 0.15575146788485472, -0.00030399289931551583, 2.7297181275762066, 0.030662836186234076, 0.1698953802286063, 0.3109214707529654, -0.0010772689742650676, 2.5782429779885416, 0.001608945791985855, -0.05580699466707014, -0.14738663799976473, -0.004277072347658766, 4.313991224922103, -0.027740725009937642],
    #     'weights_fu_with_base':[0.031164095350538913, 1.3091570298295214, -0.001450838962618584, 0.09345748918445036, 0.003130153002894097, -0.049656232497034054, 0.3415011680950051, 0.0005663657604366292, 1.3096551405180967, 0.004062318361577687, -0.0053757216649582915, 0.08477480816956491, -0.0016425210536104545, 2.3654217835530207, 0.015696195787505054, 0.20002391976879869, -0.09315550507389786, -0.002962170660641879, 4.234960103092963, -0.0022428673011072047, -0.025877130562040163, -1.9285150848642545, -0.004343849295909655, 15.799634440265821, -0.062185704912661136],
    #     'weights_cu_with_base':[-0.29637881046853853, 26.03511709887677, 0.027312294953535106, 3.0265285020597332, 0.3955898285047012, 0.8289753602934816, 13.446764776387367, -0.017450663716743864, 18.960008100517687, 0.08563325486872603, 2.023060856962742, 6.7182227010312525, -0.008135502137098856, 16.37777441440972, -0.003628978348878731, 0.9019423984257425, -2.9116321942494, 0.10756483535563985, -0.9657363577435052, 0.1543499903785995, -1.896733390678069, 51.547885705041395, -0.15163128224083874, -89.01772690925296, -0.0788002737672278]
    #     }



    weights = weights_dict[f"weights_{symbol}_with_base"]


    X = table[factor_columns].to_numpy()
    table['factor'] = np.dot(X, weights)