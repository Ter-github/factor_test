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

    # mean_dict = {'mean_ag':[0.000985, 2.5e-05, -3.7e-05, -0.000444, -0.001525],
    # 'mean_sp':[-0.002138, -7.9e-05, -3e-06, 0.003151, -0.002228],
    # 'mean_rb':[-0.001612, -0.000204, -8e-06, 0.000186, -0.003379],
    # 'mean_ru':[0.002101, -8.1e-05, -0.019705, -5e-06, 0.007246, 0.003264],
    # 'mean_hc':[-0.00185, -0.000106, -0.101609, -9e-06, 0.000614, -0.002489],
    # 'mean_fu':[0.00111, -0.00028, -5.4e-05, 7e-06, -0.003258, -0.004355],
    # 'mean_cu':[-0.002391, -1.2e-05, 0.005049, -0.000108, -0.001333, -0.001888]}

    # std_dict = {'std_ag':[0.506593, 0.262882, 0.234648, 1.164534, 1.196104],
    # 'std_sp':[0.395065, 0.233358, 0.146598, 1.250993, 1.299763],
    # 'std_rb':[0.495289, 0.390561, 0.204661, 1.244451, 1.269529],
    # 'std_ru':[0.379999, 0.212627, 28.503034, 0.141854, 1.245817, 1.297983],
    # 'std_hc':[0.547106, 0.319683, 60.892698, 0.22018, 1.231242, 1.238514],
    # 'std_fu':[0.416293, 0.258161, 42.914727, 0.167783, 1.228708, 1.260284],
    # 'std_cu':[0.557755, 0.192296, 11.99653, 0.264585, 1.15612, 1.137496]}


    # mean = mean_dict[f'mean_{symbol}']
    # std = std_dict[f'std_{symbol}']

    # table[factor_columns] = std_factor(table[factor_columns],mean,std)
    # table[factor_columns].fillna(0,inplace = True)


    # 五因子模型（不加窗口期）
    # weights_dict = {
    #     'weights_ag_with_base':[0.1667287657251836, 1.3609398279149973, 0.0011721621496397383, 0.8631162331726975, 0.014854453744628708],
    #     'weights_sp_with_base':[0.4190766823288031, 2.9166041117431782, -0.004754504338588828, 3.2195925979878086, 0.0234058616366879],
    #     'weights_rb_with_base':[0.23327073512682828, 0.6009786133673489, -0.00016644581085667476, 1.7935429623386243, 0.014706154006853805],
    #     'weights_ru_with_base':[1.115447959320505, 9.460848168723894, -0.014446352533269015, 7.630875120111201, 0.06270628156294521],
    #     'weights_hc_with_base':[0.25460389562441876, 0.9852656745679826, -0.0007161654431703774, 1.4566264912518003, 0.006318077093739149],
    #     'weights_fu_with_base':[0.2447690440146759, 1.1700526587455888, -0.0015991526842266681, 1.792758643115566, -0.00430222507218492],
    #     'weights_cu_with_base':[1.6876079018067431, 34.725403170495845, 0.0466512021191831, 1.4976080608055327, 0.13926651298799103]
    #     }

    # 五因子模型（单窗口期：[10,]）
    weights_dict = {
        'weights_ag_with_base':[0.07289214146495997, 0.6191242974014147, 0.0014106835116773407, 0.5267476377970775, 0.016110554919328297],
        'weights_sp_with_base':[0.2443152024109572, 2.153498771437588, -0.003926487000868565, 2.9449523722829456, 0.0057014163450056264],
        'weights_rb_with_base':[0.1433218162645508, 0.4727344727946159, -0.0003144407846917465, 1.3981607090480401, 0.011043704679788185],
        'weights_ru_with_base':[0.726491829642686, 7.624400183558131, -0.011495354077673946, 7.810809955430463, 0.00297793104136111],
        'weights_hc_with_base':[0.1597999413349647, 0.7887111138211873, -0.0008999174885492025, 1.4087708365352116, -0.0047795387875013695],
        'weights_fu_with_base':[0.14680601681521033, 0.5605370239103893, -0.0009163492466684386, 2.2405198382544125, -0.012588517527250537],
        'weights_cu_with_base':[0.7072433205828301, 25.62021039307153, 0.05519970889572648, -1.3626295429483453, 0.04874270693280487]
        }

    # # 五因子模型（多窗口期：[10,26,40,60,120]）
    # weights_dict = {
    #     'weights_ag_with_base':[-0.6396554882852781, 22.253634119619523, 0.02557680223038756, 6.180723252721563, 0.2406336044104225, 1.401066194290231, 2.6773355244914554, -0.05374589761312223, 24.810935301753553, 0.12425760428760163, 2.8581712608490206, -0.4689789691669593, 0.046064693187394735, 3.8579606711137724, -0.2121088739063698, -0.6903135939349665, -14.830210615958467, 0.20106455713946905, -31.95226026811632, 0.41806671309642696, -2.59446086719592, 46.032620512199536, -0.05206817499860075, -125.74806745331595, -0.09974847669035959],
    #     'weights_sp_with_base':[-0.07897066960877916, 1.9612778239688589, -0.005541089433704343, 3.5322664921254714, 0.023158476168273084, 0.38565231519149895, -0.5182541541853354, 0.001665288547488826, 4.89516083310901, 0.021372026672508412, 0.3421881914515589, 0.45239267543097283, -0.0007213520318455504, -0.563102416630214, -0.012655630249167767, -0.2052581017975333, 0.2808423481735511, -0.00029433146291547787, -5.036515617293979, 0.034123666824693866, -0.27570567987860084, 1.451838521762482, 0.004579170895167446, -3.369249142885913, -0.23281928340752558],
    #     'weights_rb_with_base':[-0.22457230471801812, 0.4469437157021639, -0.0001105595500312252, 0.9330148342452514, 0.014188910293741147, 0.017266134886883326, -0.0716488186341231, -0.0003447555619218872, 4.133863141613971, 0.01768274221935523, 0.30968561920754994, 0.04645034131489181, -0.0001813008871386098, 3.120148348130875, 0.007169736589483173, 0.12097301507413025, 0.03336147174901658, -0.0002850107311006592, 0.2272334510329965, 0.019549837397713318, -0.12037143040778196, 0.1155528929201957, -0.0022067093696915272, 0.23927883560927957, -0.0007479812704870313],
    #     'weights_ru_with_base':[-1.4455400950487107, 5.858687269782112, -0.010249392220742767, 2.309159533619505, 0.06823602581474604, 0.11529824357941892, -0.729753236033245, -0.003170855743157579, 21.28839620035281, 0.01680985144598017, 1.156778635695271, 1.9300347417503678, -0.0111167415788446, 18.798389506590645, 0.06138553183282395, 0.6959276831050553, 2.094562819935419, -0.014479624874174679, 12.24458356255848, -0.004579916174003901, 0.3374424527209604, 7.283401471711275, -0.03812638262868906, 22.686542415616355, -0.459363947977767],
    #     'weights_hc_with_base':[-0.21041873567633265, 0.6201935580839905, -0.0005028136907940098, 0.9852348682291905, 0.00825613850178011, 0.04429390754465387, 0.01781886603270545, -0.0012400905939389687, 4.130273611689439, 0.00819054788425659, 0.2916009692347561, 0.1485771073343093, -0.00026290941860995413, 2.6210603345787846, 0.014681322499773573, 0.09490595510637345, 0.22831358879109673, -0.0005197665391966097, 0.33510714559756144, 0.012217909035379537, -0.08956386706389106, -0.08612410823322518, -0.003308050678098699, 1.4340039351579472, -0.02777125569851752],
    #     'weights_fu_with_base':[-0.14790208410366185, 0.8957947356038723, -0.0010248693629990044, 0.3531352781364786, 0.0019366155855214467, 0.04262925140460978, -0.04484510445130185, 0.000820909621585765, 1.4616631311782464, 0.010462834494095014, -0.02802983474147358, 0.044900851822442744, -0.0017574631250652862, 1.8256119695906394, 0.008227187262079366, 0.18818688836885714, 0.006555393577854865, -0.002862923197957191, 4.728605070867054, -0.0005175824951177978, 0.016875218610587488, -1.725712959657126, -0.004493383559419972, 15.989282106708055, -0.054118388812971416],
    #     'weights_cu_with_base':[-0.6396554882852781, 22.253634119619523, 0.02557680223038756, 6.180723252721563, 0.2406336044104225, 1.401066194290231, 2.6773355244914554, -0.05374589761312223, 24.810935301753553, 0.12425760428760163, 2.8581712608490206, -0.4689789691669593, 0.046064693187394735, 3.8579606711137724, -0.2121088739063698, -0.6903135939349665, -14.830210615958467, 0.20106455713946905, -31.95226026811632, 0.41806671309642696, -2.59446086719592, 46.032620512199536, -0.05206817499860075, -125.74806745331595, -0.09974847669035959]
    #     }




    weights = weights_dict[f"weights_{symbol}_with_base"]


    X = table[factor_columns].to_numpy()
    table['factor'] = np.dot(X, weights)