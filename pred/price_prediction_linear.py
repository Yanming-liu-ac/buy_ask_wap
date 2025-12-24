"""
订单薄数据预测 - 线性回归方案

【核心思想】
使用历史订单薄数据（价格、流动性、价差等）预测未来60秒的价格

【方法】
多元线性回归 (Multiple Linear Regression)
公式: y = w1*x1 + w2*x2 + ... + wn*xn + b
其中: y是未来价格，x1...xn是各种特征（滞后价格、成交量等）

【优点】
1. 简单：容易理解和实现
2. 快速：训练和预测都很快
3. 可解释性强：可以看到哪些特征最重要

【缺点】
1. 只能捕捉线性关系
2. 对非线性模式无能为力
3. 容易受异常值影响

【适用场景】
- 快速建立基准模型
- 理解特征重要性
- 对比复杂模型的效果
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import pandas as pd          # 数据处理
import numpy as np           # 数值计算
import matplotlib.pyplot as plt  # 可视化
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 三种线性模型
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 评估指标
from sklearn.preprocessing import StandardScaler  # 数据标准化
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息，让输出更清晰

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

print("=" * 80)
print("订单薄数据预测 - 线性回归方案")
print("=" * 80)

# ============================================================================
# 1. 数据加载
# ============================================================================
# 【目标】读取订单薄snapshot数据、订单数据(MBO)和交易数据(MBP)
# 【数据来源】
#   - orderbook: process_orderbook_data.py 生成的采样数据
#   - mbo: 订单数据（Market By Order）
#   - mbp: 交易数据（Market By Price）
# 【数据特点】
#   - 每3秒一个快照
#   - 包含bid/ask价格和数量
#   - 覆盖10:00-14:50交易时段
# ============================================================================
print("\n[1/7] 加载数据...")

# 读取parquet格式数据（比CSV快10倍+）
df = pd.read_parquet('orderbook_sampled_10min_intervals.parquet')

# 确保timestamp是datetime类型，方便后续时间操作
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 按时间排序（时间序列必须有序！）
# reset_index(drop=True): 重置索引为0,1,2...
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"[OK] 订单薄数据加载完成")
print(f"  - 总记录数: {len(df):,}")
print(f"  - 时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"  - 时间间隔: 每3秒一个snapshot")

# ========================================================================
# 加载订单数据 (MBO - Market By Order)
# ========================================================================
print("\n  加载订单数据 (MBO)...")
try:
    mbo = pd.read_csv('mbo.csv')
    mbo['ts_event'] = pd.to_datetime(mbo['ts_event'])
    # 转换为本地时间（去除时区信息）以匹配订单薄数据
    if mbo['ts_event'].dt.tz is not None:
        mbo['ts_event'] = mbo['ts_event'].dt.tz_localize(None)
    mbo = mbo.sort_values('ts_event').reset_index(drop=True)
    print(f"  [OK] MBO数据加载完成: {len(mbo):,} 条记录")
except FileNotFoundError:
    print("  [警告] mbo.csv 未找到，将跳过订单特征")
    mbo = None

# ========================================================================
# 加载交易数据 (MBP - Market By Price)
# ========================================================================
print("\n  加载交易数据 (MBP)...")
try:
    mbp = pd.read_csv('mbp.csv')
    mbp['ts_event'] = pd.to_datetime(mbp['ts_event'])
    # 转换为本地时间（去除时区信息）以匹配订单薄数据
    if mbp['ts_event'].dt.tz is not None:
        mbp['ts_event'] = mbp['ts_event'].dt.tz_localize(None)
    mbp = mbp.sort_values('ts_event').reset_index(drop=True)
    print(f"  [OK] MBP数据加载完成: {len(mbp):,} 条记录")
except FileNotFoundError:
    print("  [警告] mbp.csv 未找到，将跳过交易特征")
    mbp = None

# ============================================================================
# 2. 特征工程
# ============================================================================
# 【核心概念】特征工程是机器学习最重要的环节！
# 【原则】用历史信息预测未来，不能使用未来信息！
# 【目标】从原始数据中提取有预测能力的特征
# ============================================================================
print("\n[2/7] 特征工程...")

def create_order_trade_features(df, mbo=None, mbp=None, lookback_seconds=5):
    """
    从订单和交易数据中提取特征（完全矢量化实现）
    
    【核心思路】
    对于每个orderbook snapshot，找到对应的时间点，然后计算过去lookback_seconds秒内的：
    - 买/卖订单数量、金额
    - 买/卖成交数量、金额
    - 订单流不平衡
    - 交易强度
    
    【矢量化策略】
    使用numpy的searchsorted + 累计和，完全避免Python循环
    
    参数:
        df: 订单薄数据（包含timestamp列）
        mbo: 订单数据（Market By Order）
        mbp: 交易数据（Market By Price）
        lookback_seconds: 回看时间窗口（秒）
    
    返回:
        df: 添加了订单/交易特征的数据
    """
    df = df.copy()
    lookback = pd.Timedelta(seconds=lookback_seconds)
    
    print(f"  - 计算过去{lookback_seconds}秒的订单/交易特征...")
    
    # ========================================================================
    # 处理MBO数据（订单数据）- 完全矢量化版本
    # ========================================================================
    if mbo is not None and len(mbo) > 0:
        print("    * 处理订单数据 (MBO)...")
        
        # 过滤出买卖订单
        mbo_buy = mbo[mbo['side'] == 'B'].copy()
        mbo_sell = mbo[mbo['side'] == 'S'].copy()
        
        # 计算订单金额 = price * size
        mbo_buy['order_amount'] = mbo_buy['price'] * mbo_buy['size']
        mbo_sell['order_amount'] = mbo_sell['price'] * mbo_sell['size']
        
        # 使用merge_asof进行时间窗口聚合
        # 策略：为每个时间点创建两个merge：一个在t时刻，一个在t-lookback时刻
        
        # 准备买单数据
        mbo_buy_agg = mbo_buy.groupby('ts_event').agg({
            'size': 'sum',
            'order_amount': 'sum',
            'order_id': 'count'  # 订单数量
        }).reset_index()
        mbo_buy_agg.columns = ['ts_event', 'buy_volume', 'buy_amount', 'buy_count']
        mbo_buy_agg = mbo_buy_agg.sort_values('ts_event')
        
        # 计算累计和
        mbo_buy_agg['buy_volume_cumsum'] = mbo_buy_agg['buy_volume'].cumsum()
        mbo_buy_agg['buy_amount_cumsum'] = mbo_buy_agg['buy_amount'].cumsum()
        mbo_buy_agg['buy_count_cumsum'] = mbo_buy_agg['buy_count'].cumsum()
        
        # 准备卖单数据
        mbo_sell_agg = mbo_sell.groupby('ts_event').agg({
            'size': 'sum',
            'order_amount': 'sum',
            'order_id': 'count'
        }).reset_index()
        mbo_sell_agg.columns = ['ts_event', 'sell_volume', 'sell_amount', 'sell_count']
        mbo_sell_agg = mbo_sell_agg.sort_values('ts_event')
        
        # 计算累计和
        mbo_sell_agg['sell_volume_cumsum'] = mbo_sell_agg['sell_volume'].cumsum()
        mbo_sell_agg['sell_amount_cumsum'] = mbo_sell_agg['sell_amount'].cumsum()
        mbo_sell_agg['sell_count_cumsum'] = mbo_sell_agg['sell_count'].cumsum()
        
        # 为df添加lookback时间列
        df['timestamp_start'] = df['timestamp'] - lookback
        
        # 使用merge_asof获取t时刻的累计值
        df = pd.merge_asof(df, mbo_buy_agg[['ts_event', 'buy_volume_cumsum', 'buy_amount_cumsum', 'buy_count_cumsum']],
                          left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
        
        # 使用merge_asof获取t-lookback时刻的累计值
        df = pd.merge_asof(df, mbo_buy_agg[['ts_event', 'buy_volume_cumsum', 'buy_amount_cumsum', 'buy_count_cumsum']],
                          left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
        
        # 计算区间内的增量 = cumsum(t) - cumsum(t-lookback)
        df[f'buy_order_volume_{lookback_seconds}s'] = (df['buy_volume_cumsum_t'] - df['buy_volume_cumsum_t0']).fillna(0)
        df[f'buy_order_amount_{lookback_seconds}s'] = (df['buy_amount_cumsum_t'] - df['buy_amount_cumsum_t0']).fillna(0)
        df[f'buy_order_count_{lookback_seconds}s'] = (df['buy_count_cumsum_t'] - df['buy_count_cumsum_t0']).fillna(0)
        
        # 同样处理卖单
        # 删除之前的ts_event列避免冲突
        if 'ts_event' in df.columns:
            df = df.drop(columns=['ts_event'])
        
        df = pd.merge_asof(df, mbo_sell_agg[['ts_event', 'sell_volume_cumsum', 'sell_amount_cumsum', 'sell_count_cumsum']],
                          left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
        
        df = pd.merge_asof(df, mbo_sell_agg[['ts_event', 'sell_volume_cumsum', 'sell_amount_cumsum', 'sell_count_cumsum']],
                          left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
        
        df[f'sell_order_volume_{lookback_seconds}s'] = (df['sell_volume_cumsum_t'] - df['sell_volume_cumsum_t0']).fillna(0)
        df[f'sell_order_amount_{lookback_seconds}s'] = (df['sell_amount_cumsum_t'] - df['sell_amount_cumsum_t0']).fillna(0)
        df[f'sell_order_count_{lookback_seconds}s'] = (df['sell_count_cumsum_t'] - df['sell_count_cumsum_t0']).fillna(0)
        
        # 删除临时列
        temp_cols = [c for c in df.columns if '_cumsum' in c or c == 'ts_event']
        df = df.drop(columns=temp_cols, errors='ignore')
        
        # 计算订单流不平衡
        df[f'order_flow_imbalance_{lookback_seconds}s'] = (
            (df[f'buy_order_volume_{lookback_seconds}s'] - df[f'sell_order_volume_{lookback_seconds}s']) /
            (df[f'buy_order_volume_{lookback_seconds}s'] + df[f'sell_order_volume_{lookback_seconds}s'] + 1e-8)
        )
        
        df[f'order_amount_imbalance_{lookback_seconds}s'] = (
            (df[f'buy_order_amount_{lookback_seconds}s'] - df[f'sell_order_amount_{lookback_seconds}s']) /
            (df[f'buy_order_amount_{lookback_seconds}s'] + df[f'sell_order_amount_{lookback_seconds}s'] + 1e-8)
        )
        
        print(f"      ✓ 订单特征: 买单数={df[f'buy_order_count_{lookback_seconds}s'].mean():.1f}, "
              f"卖单数={df[f'sell_order_count_{lookback_seconds}s'].mean():.1f}")
    
    # ========================================================================
    # 处理MBP数据（交易数据）- 完全矢量化版本
    # ========================================================================
    if mbp is not None and len(mbp) > 0:
        print("    * 处理交易数据 (MBP)...")
        
        # MBP包含了成交信息
        # 过滤出成交记录（如果有action字段）
        if 'action' in mbp.columns:
            trades = mbp[mbp['action'] == 'T'].copy()
        else:
            trades = mbp.copy()
        
        if len(trades) > 0 and 'price' in trades.columns and 'size' in trades.columns:
            # 计算成交金额
            trades['trade_amount'] = trades['price'] * trades['size']
            
            # 分离买卖方向
            if 'side' in trades.columns:
                trades_buy = trades[trades['side'] == 'B'].copy()
                trades_sell = trades[trades['side'] == 'S'].copy()
            else:
                # 如果没有side字段，假设都是买方
                trades_buy = trades.copy()
                trades_sell = pd.DataFrame()
            
            # 处理买方成交
            if len(trades_buy) > 0:
                trades_buy_agg = trades_buy.groupby('ts_event').agg({
                    'size': 'sum',
                    'trade_amount': 'sum',
                    'sequence': 'count'  # 成交笔数
                }).reset_index()
                trades_buy_agg.columns = ['ts_event', 'buy_trade_volume', 'buy_trade_amount', 'buy_trade_count']
                trades_buy_agg = trades_buy_agg.sort_values('ts_event')
                
                # 累计和
                trades_buy_agg['buy_trade_volume_cumsum'] = trades_buy_agg['buy_trade_volume'].cumsum()
                trades_buy_agg['buy_trade_amount_cumsum'] = trades_buy_agg['buy_trade_amount'].cumsum()
                trades_buy_agg['buy_trade_count_cumsum'] = trades_buy_agg['buy_trade_count'].cumsum()
                
                # merge
                if 'timestamp_start' not in df.columns:
                    df['timestamp_start'] = df['timestamp'] - lookback
                
                df = pd.merge_asof(df, trades_buy_agg[['ts_event', 'buy_trade_volume_cumsum', 'buy_trade_amount_cumsum', 'buy_trade_count_cumsum']],
                                  left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
                
                df = pd.merge_asof(df, trades_buy_agg[['ts_event', 'buy_trade_volume_cumsum', 'buy_trade_amount_cumsum', 'buy_trade_count_cumsum']],
                                  left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
                
                df[f'buy_trade_volume_{lookback_seconds}s'] = (df['buy_trade_volume_cumsum_t'] - df['buy_trade_volume_cumsum_t0']).fillna(0)
                df[f'buy_trade_amount_{lookback_seconds}s'] = (df['buy_trade_amount_cumsum_t'] - df['buy_trade_amount_cumsum_t0']).fillna(0)
                df[f'buy_trade_count_{lookback_seconds}s'] = (df['buy_trade_count_cumsum_t'] - df['buy_trade_count_cumsum_t0']).fillna(0)
            else:
                df[f'buy_trade_volume_{lookback_seconds}s'] = 0
                df[f'buy_trade_amount_{lookback_seconds}s'] = 0
                df[f'buy_trade_count_{lookback_seconds}s'] = 0
            
            # 处理卖方成交
            if len(trades_sell) > 0:
                trades_sell_agg = trades_sell.groupby('ts_event').agg({
                    'size': 'sum',
                    'trade_amount': 'sum',
                    'sequence': 'count'
                }).reset_index()
                trades_sell_agg.columns = ['ts_event', 'sell_trade_volume', 'sell_trade_amount', 'sell_trade_count']
                trades_sell_agg = trades_sell_agg.sort_values('ts_event')
                
                trades_sell_agg['sell_trade_volume_cumsum'] = trades_sell_agg['sell_trade_volume'].cumsum()
                trades_sell_agg['sell_trade_amount_cumsum'] = trades_sell_agg['sell_trade_amount'].cumsum()
                trades_sell_agg['sell_trade_count_cumsum'] = trades_sell_agg['sell_trade_count'].cumsum()
                
                # 删除之前的ts_event列避免冲突
                if 'ts_event' in df.columns:
                    df = df.drop(columns=['ts_event'])
                
                df = pd.merge_asof(df, trades_sell_agg[['ts_event', 'sell_trade_volume_cumsum', 'sell_trade_amount_cumsum', 'sell_trade_count_cumsum']],
                                  left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
                
                df = pd.merge_asof(df, trades_sell_agg[['ts_event', 'sell_trade_volume_cumsum', 'sell_trade_amount_cumsum', 'sell_trade_count_cumsum']],
                                  left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
                
                df[f'sell_trade_volume_{lookback_seconds}s'] = (df['sell_trade_volume_cumsum_t'] - df['sell_trade_volume_cumsum_t0']).fillna(0)
                df[f'sell_trade_amount_{lookback_seconds}s'] = (df['sell_trade_amount_cumsum_t'] - df['sell_trade_amount_cumsum_t0']).fillna(0)
                df[f'sell_trade_count_{lookback_seconds}s'] = (df['sell_trade_count_cumsum_t'] - df['sell_trade_count_cumsum_t0']).fillna(0)
            else:
                df[f'sell_trade_volume_{lookback_seconds}s'] = 0
                df[f'sell_trade_amount_{lookback_seconds}s'] = 0
                df[f'sell_trade_count_{lookback_seconds}s'] = 0
            
            # 删除临时列
            temp_cols = [c for c in df.columns if '_cumsum' in c or c == 'ts_event']
            df = df.drop(columns=temp_cols, errors='ignore')
            
            # 计算成交不平衡
            df[f'trade_volume_imbalance_{lookback_seconds}s'] = (
                (df[f'buy_trade_volume_{lookback_seconds}s'] - df[f'sell_trade_volume_{lookback_seconds}s']) /
                (df[f'buy_trade_volume_{lookback_seconds}s'] + df[f'sell_trade_volume_{lookback_seconds}s'] + 1e-8)
            )
            
            df[f'trade_amount_imbalance_{lookback_seconds}s'] = (
                (df[f'buy_trade_amount_{lookback_seconds}s'] - df[f'sell_trade_amount_{lookback_seconds}s']) /
                (df[f'buy_trade_amount_{lookback_seconds}s'] + df[f'sell_trade_amount_{lookback_seconds}s'] + 1e-8)
            )
            
            # 交易强度（总成交量/时间）
            df[f'trade_intensity_{lookback_seconds}s'] = (
                (df[f'buy_trade_volume_{lookback_seconds}s'] + df[f'sell_trade_volume_{lookback_seconds}s']) / 
                lookback_seconds
            )
            
            print(f"      ✓ 成交特征: 买方成交={df[f'buy_trade_count_{lookback_seconds}s'].mean():.1f}, "
                  f"卖方成交={df[f'sell_trade_count_{lookback_seconds}s'].mean():.1f}")
    
    # 清理timestamp_start列
    if 'timestamp_start' in df.columns:
        df = df.drop(columns=['timestamp_start'])
    
    return df

def create_features(df):
    """
    创建预测特征
    
    【思路】
    机器学习模型不能直接看到"过去的价格序列"，需要我们把序列信息
    转换成数值特征。就像告诉模型：
    - "3秒前价格是多少"（滞后特征）
    - "过去1分钟平均价格是多少"（滚动统计）
    - "价格在上涨还是下跌"（变化特征）
    
    参数:
        df: 原始订单薄数据
    
    返回:
        df: 添加了特征列的数据
    """
    df = df.copy()  # 复制一份，不修改原数据
    
    # ========================================================================
    # 特征类型1: 滞后特征 (Lag Features)
    # ========================================================================
    # 【定义】过去某个时间点的值
    # 【用途】捕捉时间序列的自相关性（历史价格影响未来价格）
    # 【例子】如果过去1分钟价格一直在涨，未来可能继续涨
    # ========================================================================
    print("  - 创建滞后特征...")
    for lag in [1, 2, 3, 5, 10, 20]:  # lag=1表示3秒前，lag=20表示60秒前
        # 价格滞后：过去N个时间步的价格
        df[f'mid_price_lag_{lag}'] = df['mid_price'].shift(lag)
        
        # 流动性滞后：过去N个时间步的买/卖量
        # 重要！流动性预示价格压力
        df[f'bid1_v_lag_{lag}'] = df['bid1_v'].shift(lag)
        df[f'ask1_v_lag_{lag}'] = df['ask1_v'].shift(lag)
        
        # 价差滞后：过去N个时间步的买卖价差
        df[f'spread_lag_{lag}'] = df['spread'].shift(lag)
    
    # ========================================================================
    # 特征类型2: 价格变化特征 (Difference Features)
    # ========================================================================
    # 【定义】价格的一阶/二阶差分
    # 【用途】捕捉价格的动量(momentum)和趋势
    # 【为什么重要】价格变化比绝对价格更有预测性！
    # 【例子】如果过去15秒涨了0.5元，可能还会继续涨
    # ========================================================================
    print("  - 创建价格变化特征...")
    
    # 绝对变化量
    df['price_change_1'] = df['mid_price'].diff(1)    # 3秒变化
    df['price_change_5'] = df['mid_price'].diff(5)    # 15秒变化
    df['price_change_20'] = df['mid_price'].diff(20)  # 60秒变化
    
    # 百分比变化（相对变化，消除价格水平影响）
    df['price_pct_change_1'] = df['mid_price'].pct_change(1)
    df['price_pct_change_5'] = df['mid_price'].pct_change(5)
    
    # ========================================================================
    # 特征类型3: 滚动统计特征 (Rolling Statistics)
    # ========================================================================
    # 【定义】滑动窗口内的统计量（均值、标准差等）
    # 【用途】捕捉局部趋势和波动性
    # 【例子】过去30秒均价=99.5，当前价=99.8，说明价格偏高，可能回落
    # ========================================================================
    print("  - 创建滚动统计特征...")
    for window in [5, 10, 20]:  # 窗口大小：15秒、30秒、60秒
        # 移动平均：趋势指标
        df[f'price_mean_{window}'] = df['mid_price'].rolling(window).mean()
        
        # 移动标准差：波动性指标
        # 标准差大 = 波动剧烈 = 风险高
        df[f'price_std_{window}'] = df['mid_price'].rolling(window).std()
        
        # 价差均值：交易成本指标
        df[f'spread_mean_{window}'] = df['spread'].rolling(window).mean()
    
    # ========================================================================
    # 特征类型4: 订单薄不平衡特征 (Order Book Imbalance)
    # ========================================================================
    # 【定义】买卖盘力量对比
    # 【用途】捕捉市场微观结构，预测短期价格压力
    # 【理论】买压大→价格上涨，卖压大→价格下跌
    # 【非常重要】这是订单薄数据独有的信息！
    # ========================================================================
    print("  - 创建订单薄特征...")
    
    # 成交量不平衡度：范围[-1, 1]
    # +1 = 全是买压（bid_v >> ask_v）→ 价格可能上涨
    # -1 = 全是卖压（ask_v >> bid_v）→ 价格可能下跌
    #  0 = 买卖平衡
    df['volume_imbalance'] = (df['bid1_v'] - df['ask1_v']) / (df['bid1_v'] + df['ask1_v'] + 1e-8)
    df['volume_imbalance_lag_1'] = df['volume_imbalance'].shift(1)
    
    # 总流动性：买一量 + 卖一量
    # 流动性高 → 大单容易成交，价格稳定
    # 流动性低 → 大单冲击大，价格波动
    df['total_liquidity'] = df['bid1_v'] + df['ask1_v']
    df['total_liquidity_lag_1'] = df['total_liquidity'].shift(1)
    
    # ========================================================================
    # 特征类型5: 波动率特征 (Volatility)
    # ========================================================================
    # 【定义】价格波动的相对强度
    # 【用途】风险指标，预测难度指标
    # 【例子】波动率高 → 预测难 → 减少交易 / 放宽止损
    # ========================================================================
    print("  - 创建波动率特征...")
    
    # 变异系数 (CV) = 标准差 / 均值
    # 相对波动率，消除价格水平影响
    df['volatility_20'] = df['mid_price'].rolling(20).std() / (df['mid_price'].rolling(20).mean() + 1e-8)
    
    # ========================================================================
    # 特征类型6: 时间特征 (Temporal Features)
    # ========================================================================
    # 【定义】时间相关的特征
    # 【用途】捕捉日内模式（开盘/收盘效应）
    # 【例子】
    #   - 开盘前30分钟：波动大，流动性差
    #   - 午间：交易清淡
    #   - 收盘前30分钟：流动性增加，可能有强平
    # ========================================================================
    print("  - 创建时间特征...")
    
    df['hour'] = df['timestamp'].dt.hour        # 小时 (9-14)
    df['minute'] = df['timestamp'].dt.minute    # 分钟 (0-59)
    
    # 距离开盘的秒数
    df['seconds_since_open'] = (df['timestamp'].dt.hour - 9) * 3600 + df['timestamp'].dt.minute * 60
    
    # 是否接近收盘（最后30分钟）
    # 收盘前市场行为不同：抢筹/强平/流动性变化
    df['near_close'] = ((df['hour'] == 14) & (df['minute'] >= 30)).astype(int)
    
    return df

# 执行特征工程
print("\n  阶段1: 创建订单薄基础特征...")
df_features = create_features(df)

# 添加订单和交易特征
if mbo is not None or mbp is not None:
    print("\n  阶段2: 创建订单/交易特征...")
    df_features = create_order_trade_features(df_features, mbo, mbp, lookback_seconds=5)

print(f"\n[OK] 特征工程完成，共 {df_features.shape[1]} 列")

# ============================================================================
# 3. 创建预测目标
# ============================================================================
# 【目标】定义我们要预测什么
# 【重要】shift(-N) 表示"未来N步的值"，这是监督学习的标签(label)
# 【时间对齐】
#   当前行的特征 X → 预测 → 未来N步的目标 y
#   例如：10:00:00的特征 → 预测 → 10:01:00的价格
# ============================================================================
print("\n[3/7] 创建预测目标...")

# 预测时间跨度：60秒 = 20个时间步 (每步3秒)
forecast_horizon = 20

# ========================================================================
# 创建多个预测目标（可以选择预测哪个）
# ========================================================================

# 目标1: 未来60秒的价格（本脚本使用这个）
# shift(-20): 把价格往上移20行 → 当前行看到的是"未来"的价格
df_features['target_mid_price'] = df_features['mid_price'].shift(-forecast_horizon)

# 目标2: 未来60秒的价差（可用于评估交易成本）
df_features['target_spread'] = df_features['spread'].shift(-forecast_horizon)

# 目标3: 未来60秒的流动性（非常重要！影响TWAP执行）
df_features['target_bid1_v'] = df_features['bid1_v'].shift(-forecast_horizon)
df_features['target_ask1_v'] = df_features['ask1_v'].shift(-forecast_horizon)

# ========================================================================
# 删除缺失值
# ========================================================================
# 【为什么有缺失值？】
# 1. 前20行：没有足够的历史数据做滞后特征
# 2. 后20行：没有未来数据做预测目标
# 3. 滚动统计在窗口不足时也会产生NaN
#
# 【解决方案】dropna() 删除所有含NaN的行
df_ml = df_features.dropna().reset_index(drop=True)

print(f"[OK] 预测目标创建完成，有效样本数: {len(df_ml):,}")
print(f"     (删除了 {len(df_features) - len(df_ml)} 个含缺失值的样本)")

# ============================================================================
# 4. 数据划分
# ============================================================================
# 【核心原则】时间序列必须按时间顺序划分，不能随机打乱！
# 【原因】
#   1. 避免"未来信息泄露"：训练集不能包含测试集之后的数据
#   2. 模拟真实场景：训练用历史，预测用未来
#   3. 如果随机划分，模型会学到未来信息，测试效果虚高
#
# 【划分策略】
#   训练集(70%): 用于学习模型参数
#   验证集(15%): 用于调参、选模型
#   测试集(15%): 最终评估，模拟真实表现
# ============================================================================
print("\n[4/7] 数据划分...")

# ========================================================================
# 步骤1: 选择特征列
# ========================================================================
# 从所有列中筛选出特征列（排除目标列和元数据）
feature_cols = [c for c in df_ml.columns 
                if ('lag' in c or           # 滞后特征
                    'mean' in c or          # 均值特征
                    'std' in c or           # 标准差特征
                    'imbalance' in c or     # 不平衡特征
                    'liquidity' in c or     # 流动性特征
                    'volatility' in c or    # 波动率特征
                    'change' in c or        # 变化特征
                    'order' in c or         # 订单特征
                    'trade' in c or         # 交易特征
                    'intensity' in c or     # 交易强度特征
                    c in ['hour', 'minute', 'seconds_since_open', 'near_close'])  # 时间特征
                and 'target' not in c]      # 排除目标列

print(f"  - 特征数量: {len(feature_cols)}")
print(f"  - 前10个特征: {feature_cols[:10]}")
if len(feature_cols) > 10:
    # 显示订单/交易特征
    order_trade_features = [c for c in feature_cols if 'order' in c or 'trade' in c]
    if order_trade_features:
        print(f"  - 订单/交易特征 ({len(order_trade_features)}个): {order_trade_features[:5]}...")

# ========================================================================
# 步骤2: 按时间顺序划分数据集
# ========================================================================
n = len(df_ml)
train_size = int(n * 0.7)   # 70%训练
val_size = int(n * 0.15)    # 15%验证

# 时间线: |----训练集----|--验证集--|--测试集--|
#         0          train_size  train_size+val_size  n
train_df = df_ml.iloc[:train_size]                           # 前70%
val_df = df_ml.iloc[train_size:train_size+val_size]        # 中间15%
test_df = df_ml.iloc[train_size+val_size:]                  # 后15%

# ========================================================================
# 步骤3: 准备X(特征)和y(目标)
# ========================================================================
# X: 特征矩阵，每行是一个样本，每列是一个特征
# y: 目标向量，每个元素是对应样本的标签

X_train = train_df[feature_cols]          # 训练特征
y_train = train_df['target_mid_price']    # 训练目标（未来60秒价格）

X_val = val_df[feature_cols]              # 验证特征
y_val = val_df['target_mid_price']        # 验证目标

X_test = test_df[feature_cols]            # 测试特征
y_test = test_df['target_mid_price']      # 测试目标

print(f"[OK] 数据划分完成:")
print(f"  - 训练集: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
print(f"  - 验证集: {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
print(f"  - 测试集: {len(test_df):,} ({len(test_df)/n*100:.1f}%)")

# ============================================================================
# 5. 数据标准化（线性回归建议标准化）
# ============================================================================
# 【为什么要标准化？】
# 1. 特征量纲不同：价格~100，成交量~10000，时间~14，量级差异大
# 2. 线性回归对量纲敏感：大量级特征会主导模型，小量级特征被忽略
# 3. 加速收敛：梯度下降法在标准化后收敛更快
# 4. 便于解释系数：标准化后系数大小直接反映特征重要性
#
# 【标准化公式】
#   x_scaled = (x - mean) / std
#   结果：均值=0，标准差=1
#
# 【注意】
#   - 只用训练集计算mean和std（避免数据泄露）
#   - 验证集和测试集用训练集的mean和std
# ============================================================================
print("\n[5/7] 数据标准化...")

# 创建标准化器
scaler = StandardScaler()

# fit_transform(): 计算训练集的均值和标准差，并标准化
X_train_scaled = scaler.fit_transform(X_train)

# transform(): 用训练集的均值和标准差标准化验证集和测试集
# 重要：不能用验证/测试集自己的统计量！
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("[OK] 标准化完成（均值=0，标准差=1）")
print(f"     示例: 第1个特征原始范围 [{X_train.iloc[:, 0].min():.2f}, {X_train.iloc[:, 0].max():.2f}]")
print(f"          标准化后范围 [{X_train_scaled[:, 0].min():.2f}, {X_train_scaled[:, 0].max():.2f}]")

# ============================================================================
# 6. 训练三种线性模型
# ============================================================================
# 【线性回归家族】
# 基础公式: y = w1*x1 + w2*x2 + ... + wn*xn + b
# 
# 三种变体的区别在于如何防止过拟合：
# 
# 1. 普通线性回归 (OLS): 无正则化
#    优化目标: min Σ(y_true - y_pred)²
#    优点: 简单，可解释性强
#    缺点: 容易过拟合，特征多时不稳定
#
# 2. Ridge回归 (L2正则化): 惩罚系数的平方
#    优化目标: min Σ(y_true - y_pred)² + α*Σw²
#    优点: 所有特征保留，系数缩小，稳定性好
#    缺点: 不能做特征选择
#
# 3. Lasso回归 (L1正则化): 惩罚系数的绝对值
#    优化目标: min Σ(y_true - y_pred)² + α*Σ|w|
#    优点: 自动特征选择（不重要的特征系数变0）
#    缺点: 可能丢失有用信息
# ============================================================================
print("\n[6/7] 训练线性模型...")

models = {}  # 存储所有模型

# ========================================================================
# 模型1: 普通线性回归 (Ordinary Least Squares)
# ========================================================================
# 【方法】最小二乘法
# 【适用】特征少、样本多、线性关系强
# 【参数】无超参数
print("\n  [模型1] 普通线性回归 (OLS)...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)  # 学习 w 和 b
models['Linear Regression'] = lr
print("  [OK] 训练完成")
print(f"       学到了 {len(lr.coef_)} 个系数")

# ========================================================================
# 模型2: Ridge回归（岭回归）
# ========================================================================
# 【方法】L2正则化
# 【适用】特征多、有共线性、需要稳定性
# 【参数】alpha: 正则化强度
#   - alpha=0: 等于普通线性回归
#   - alpha越大: 系数越小，偏差增加但方差减小
#   - 典型范围: 0.01 ~ 100
print("\n  [模型2] Ridge回归 (L2正则化)...")
ridge = Ridge(alpha=1.0)  # alpha=1.0 是常用起始值
ridge.fit(X_train_scaled, y_train)
models['Ridge'] = ridge
print("  [OK] 训练完成")
print(f"       正则化强度 alpha = {ridge.alpha}")

# ========================================================================
# 模型3: Lasso回归
# ========================================================================
# 【方法】L1正则化
# 【适用】特征多、想做特征选择、追求稀疏性
# 【参数】alpha: 正则化强度
#   - alpha=0: 等于普通线性回归
#   - alpha越大: 更多特征系数变0
#   - 典型范围: 0.001 ~ 10
print("\n  [模型3] Lasso回归 (L1正则化)...")
lasso = Lasso(alpha=0.1)  # alpha较小，保留更多特征
lasso.fit(X_train_scaled, y_train)
models['Lasso'] = lasso
print("  [OK] 训练完成")
# 统计有多少特征被保留（系数非零）
non_zero_features = np.sum(lasso.coef_ != 0)
print(f"       保留了 {non_zero_features}/{len(lasso.coef_)} 个特征")

# ============================================================================
# 7. 模型评估
# ============================================================================
# 【评估指标全解析】
# 不同指标评估不同方面，需要综合考虑
# ============================================================================
print("\n[7/7] 模型评估...")

def evaluate_model(model, X, y, dataset_name):
    """
    评估模型性能
    
    返回多个评估指标，全面评估模型质量
    
    参数:
        model: 训练好的模型
        X: 特征数据
        y: 真实目标值
        dataset_name: 数据集名称（用于打印）
    
    返回:
        dict: 包含各种评估指标的字典
    """
    # 预测
    y_pred = model.predict(X)
    
    # ========================================================================
    # 指标1: MAE (Mean Absolute Error) - 平均绝对误差
    # ========================================================================
    # 【定义】|真实值 - 预测值|的平均
    # 【解释】平均预测偏差多少
    # 【优点】直观，和目标值同单位（如果预测价格，MAE就是价格单位）
    # 【例子】MAE=0.22 表示平均预测误差 0.22 元
    mae = mean_absolute_error(y, y_pred)
    
    # ========================================================================
    # 指标2: RMSE (Root Mean Squared Error) - 均方根误差
    # ========================================================================
    # 【定义】sqrt(平均((真实值 - 预测值)²))
    # 【解释】预测误差的标准差
    # 【特点】对大误差更敏感（因为平方）
    # 【对比】RMSE > MAE 说明存在一些大的预测误差
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # ========================================================================
    # 指标3: MAPE (Mean Absolute Percentage Error) - 平均绝对百分比误差
    # ========================================================================
    # 【定义】|(真实值 - 预测值) / 真实值|的平均 × 100%
    # 【解释】相对误差，消除量级影响
    # 【优点】可以跨数据集比较
    # 【例子】MAPE=0.23% 表示平均预测偏差 0.23%
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    # ========================================================================
    # 指标4: R² (R-squared) - 决定系数
    # ========================================================================
    # 【定义】1 - (模型误差 / 基准误差)
    # 【解释】模型解释了多少比例的方差
    # 【范围】(-∞, 1]，越接近1越好
    #   - R²=1: 完美预测
    #   - R²=0: 模型等同于"预测均值"
    #   - R²<0: 模型比预测均值还差！
    # 【例子】R²=0.98 表示模型解释了98%的价格变化
    r2 = r2_score(y, y_pred)
    
    # ========================================================================
    # 指标5: 方向准确率 (Direction Accuracy)
    # ========================================================================
    # 【定义】预测的涨跌方向和真实方向一致的比例
    # 【重要性】对交易来说，方向比具体价格更重要！
    # 【基准】随机猜测 = 50%
    # 【目标】> 55% 才有交易价值
    # 【计算】比较相邻时间点的价格变化符号
    y_diff = y.diff().fillna(0).values         # 真实价格变化
    pred_diff = pd.Series(y_pred).diff().fillna(0).values  # 预测价格变化
    direction_acc = ((y_diff > 0) == (pred_diff > 0)).mean() * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'Direction_Acc': direction_acc,
        'predictions': y_pred
    }

# ========================================================================
# 评估所有模型
# ========================================================================
# 【评估策略】
# 1. 训练集：检查是否欠拟合（训练效果差→模型太简单）
# 2. 验证集：用于调参、选模型
# 3. 测试集：最终评估，代表真实表现
#
# 【过拟合判断】
# - 训练集效果好，验证/测试集效果差 → 过拟合
# - 解决：增加正则化、减少特征、增加数据
#
# 【欠拟合判断】
# - 训练集效果差 → 欠拟合
# - 解决：增加特征、使用更复杂模型
# ========================================================================
results = {}
for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    # 在三个数据集上评估
    train_res = evaluate_model(model, X_train_scaled, y_train, "训练集")
    val_res = evaluate_model(model, X_val_scaled, y_val, "验证集")
    test_res = evaluate_model(model, X_test_scaled, y_test, "测试集")
    
    # 保存结果
    results[model_name] = {
        'train': train_res,
        'val': val_res,
        'test': test_res
    }
    
    # 打印训练集结果
    print(f"\n训练集:")
    print(f"  MAE:  {train_res['MAE']:.4f}")
    print(f"  RMSE: {train_res['RMSE']:.4f}")
    print(f"  MAPE: {train_res['MAPE']:.4f}%")
    print(f"  R2:   {train_res['R²']:.4f}")
    print(f"  方向准确率: {train_res['Direction_Acc']:.2f}%")
    
    # 打印验证集结果
    print(f"\n验证集:")
    print(f"  MAE:  {val_res['MAE']:.4f}")
    print(f"  RMSE: {val_res['RMSE']:.4f}")
    print(f"  MAPE: {val_res['MAPE']:.4f}%")
    print(f"  R2:   {val_res['R²']:.4f}")
    print(f"  方向准确率: {val_res['Direction_Acc']:.2f}%")
    
    # 打印测试集结果（最重要！）
    print(f"\n测试集:")
    print(f"  MAE:  {test_res['MAE']:.4f}")
    print(f"  RMSE: {test_res['RMSE']:.4f}")
    print(f"  MAPE: {test_res['MAPE']:.4f}%")
    print(f"  R2:   {test_res['R²']:.4f}")
    print(f"  方向准确率: {test_res['Direction_Acc']:.2f}%")

# ============================================================================
# 8. 可视化对比
# ============================================================================
# 【目的】通过图表直观对比模型性能
# 【输出】3张图：模型对比、预测结果、特征重要性
# ============================================================================
print("\n\n生成可视化结果...")

# ========================================================================
# 图1: 模型性能对比（4个指标的分组柱状图）
# ========================================================================
# 【作用】一眼看出哪个模型最好，是否过拟合
# 【解读】
#   - 训练集 >> 测试集：过拟合
#   - 训练集 ≈ 测试集：泛化能力好
#   - MAE/RMSE越小越好，R²越大越好
# ========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2子图

metrics = ['MAE', 'RMSE', 'MAPE', 'R²']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]  # 定位到对应子图
    
    # 提取每个模型在三个数据集上的指标值
    model_names = list(results.keys())
    train_vals = [results[m]['train'][metric] for m in model_names]
    val_vals = [results[m]['val'][metric] for m in model_names]
    test_vals = [results[m]['test'][metric] for m in model_names]
    
    # 绘制分组柱状图
    x = np.arange(len(model_names))
    width = 0.25  # 每组柱子的宽度
    
    ax.bar(x - width, train_vals, width, label='训练集', alpha=0.8)
    ax.bar(x, val_vals, width, label='验证集', alpha=0.8)
    ax.bar(x + width, test_vals, width, label='测试集', alpha=0.8)
    
    # 美化图表
    ax.set_xlabel('模型')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} 对比')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('linear_model_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] 保存: linear_model_comparison.png")

# ========================================================================
# 图2: 预测结果可视化（使用最佳模型）
# ========================================================================
# 【目的】可视化预测效果，发现问题
# 【包含】3个子图：预测vs真实、误差序列、误差分布
# ========================================================================

# 选择测试集MAE最小的模型作为最佳模型
best_model_name = min(results.keys(), key=lambda m: results[m]['test']['MAE'])
best_model = models[best_model_name]

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# ------------------------------------------------------------------------
# 子图1: 预测值 vs 真实值（时间序列图）
# ------------------------------------------------------------------------
# 【作用】直观看出预测是否跟随真实值
# 【理想情况】两条线完全重合
# 【常见问题】
#   - 预测线太平滑：模型在做"平均"，缺乏预测力
#   - 预测线滞后：模型只是"复述"历史
#   - 预测线波动太大：过拟合
test_sample = test_df.iloc[:200]  # 只画前200个点，太多看不清
y_test_sample = y_test.iloc[:200]
y_pred_sample = results[best_model_name]['test']['predictions'][:200]

axes[0].plot(test_sample['timestamp'], y_test_sample, 
             label='真实值', alpha=0.8, linewidth=2, marker='o', markersize=2)
axes[0].plot(test_sample['timestamp'], y_pred_sample, 
             label='预测值', alpha=0.8, linewidth=2, marker='x', markersize=2)
axes[0].set_xlabel('时间')
axes[0].set_ylabel('价格')
axes[0].set_title(f'{best_model_name} - 预测结果（测试集前200个点）', 
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ------------------------------------------------------------------------
# 子图2: 预测误差时间序列
# ------------------------------------------------------------------------
# 【作用】看误差是否有模式
# 【理想情况】误差随机分布在0附近
# 【警告信号】
#   - 误差有趋势：系统性偏差
#   - 误差有周期：遗漏了某个重要特征
#   - 误差突然变大：市场状态变化
errors = y_test_sample.values - y_pred_sample
axes[1].plot(test_sample['timestamp'], errors, alpha=0.7, color='red', linewidth=1)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
axes[1].fill_between(test_sample['timestamp'], errors, 0, alpha=0.3, color='red')
axes[1].set_xlabel('时间')
axes[1].set_ylabel('预测误差（真实 - 预测）')
axes[1].set_title('预测误差时间序列', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

# ------------------------------------------------------------------------
# 子图3: 误差分布直方图
# ------------------------------------------------------------------------
# 【作用】看误差是否服从正态分布
# 【理想情况】钟形曲线，中心在0
# 【异常情况】
#   - 偏向一边：系统性高估或低估
#   - 长尾：存在极端误差
#   - 双峰：可能有两种不同的市场状态
axes[2].hist(y_test.values - results[best_model_name]['test']['predictions'], 
             bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
axes[2].set_xlabel('预测误差')
axes[2].set_ylabel('频数')
axes[2].set_title('预测误差分布', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('linear_prediction_results.png', dpi=150, bbox_inches='tight')
print("[OK] 保存: linear_prediction_results.png")

# ========================================================================
# 图3: 特征重要性（使用线性回归的系数）
# ========================================================================
# 【目的】理解哪些特征对预测最重要
# 【方法】线性回归的系数绝对值 = 特征重要性
# 【解读】
#   - 系数大：该特征对预测影响大
#   - 系数小：该特征影响小，可能可以删除
# 【应用】
#   1. 特征选择：删除不重要的特征，简化模型
#   2. 业务洞察：理解什么因素影响价格
#   3. 特征工程方向：加强重要特征的衍生特征
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 10))

# 计算特征重要性：取系数绝对值
# 为什么取绝对值？正负系数都表示影响，只是方向不同
coefficients = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': np.abs(best_model.coef_)  # 绝对值
}).sort_values('coefficient', ascending=False)

# 只展示Top 20，太多了看不清
top_20 = coefficients.head(20)

# 水平柱状图（横向），更容易阅读特征名
ax.barh(range(len(top_20)), top_20['coefficient'], alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('系数绝对值（标准化后）')
ax.set_title(f'{best_model_name} - Top 20 最重要特征', 
             fontsize=12, fontweight='bold')
ax.invert_yaxis()  # 最重要的在上面
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('linear_feature_importance.png', dpi=150, bbox_inches='tight')
print("[OK] 保存: linear_feature_importance.png")

# ============================================================================
# 9. 保存结果
# ============================================================================
# 【目的】保存训练成果，供后续使用
# 【包含】模型文件、数据表格、可视化图表
# ============================================================================
print("\n\n保存结果...")

# ========================================================================
# 保存1: 模型性能对比表（CSV格式）
# ========================================================================
# 【用途】快速对比各模型，写报告/论文时用
# 【格式】每行一个模型，每列一个指标
comparison_df = pd.DataFrame({
    '模型': list(results.keys()),
    '测试MAE': [results[m]['test']['MAE'] for m in results.keys()],
    '测试RMSE': [results[m]['test']['RMSE'] for m in results.keys()],
    '测试MAPE': [results[m]['test']['MAPE'] for m in results.keys()],
    '测试R²': [results[m]['test']['R²'] for m in results.keys()],
    '方向准确率': [results[m]['test']['Direction_Acc'] for m in results.keys()]
})
# encoding='utf-8-sig': 确保Excel能正确打开中文
comparison_df.to_csv('linear_model_comparison.csv', index=False, encoding='utf-8-sig')
print("[OK] 保存: linear_model_comparison.csv")

# ========================================================================
# 保存2: 特征重要性表（CSV格式）
# ========================================================================
# 【用途】
#   1. 特征选择：删除不重要的特征
#   2. 特征工程：重点优化重要特征
#   3. 业务解释：向他人解释模型
coefficients.to_csv('linear_feature_importance.csv', index=False, encoding='utf-8-sig')
print("[OK] 保存: linear_feature_importance.csv")

# ========================================================================
# 保存3: 最佳模型（PKL格式）
# ========================================================================
# 【用途】
#   1. 在新数据上预测
#   2. 部署到生产环境
#   3. 与其他模型集成
# 【加载方法】
#   import joblib
#   model = joblib.load('linear_regression_model.pkl')
#   scaler = joblib.load('scaler.pkl')
#   prediction = model.predict(scaler.transform(new_data))
import joblib
joblib.dump(best_model, f'{best_model_name.lower().replace(" ", "_")}_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # 标准化器也要保存！否则新数据无法预测
print(f"[OK] 保存: {best_model_name.lower().replace(' ', '_')}_model.pkl")
print("[OK] 保存: scaler.pkl")

# ============================================================================
# 10. 总结
# ============================================================================
# 【输出】训练摘要和文件清单
# 【用途】快速了解结果，知道生成了哪些文件
# ============================================================================
print("\n" + "=" * 80)
print("训练完成！")
print("=" * 80)
print(f"\n最佳模型: {best_model_name}")
print(f"测试集MAE: {results[best_model_name]['test']['MAE']:.4f}")
print(f"测试集R2: {results[best_model_name]['test']['R²']:.4f}")

print("\n生成的文件:")
print("  1. linear_model_comparison.png - 模型性能对比图")
print("  2. linear_prediction_results.png - 预测结果可视化")
print("  3. linear_feature_importance.png - 特征重要性图")
print("  4. linear_model_comparison.csv - 模型性能数据")
print("  5. linear_feature_importance.csv - 特征重要性数据")
print(f"  6. {best_model_name.lower().replace(' ', '_')}_model.pkl - 最佳模型")
print("  7. scaler.pkl - 数据标准化器")

print("\n" + "=" * 80)
print("线性回归预测完成！")
print("=" * 80)

print("\n【下一步建议】")
print("1. 查看特征重要性图，了解哪些特征最有用")
print("2. 如果方向准确率<55%，考虑改用分类模型")
print("3. 尝试预测更短的时间窗口（15秒、30秒）")
print("4. 添加更强的特征（订单薄深度、技术指标）")
print("5. 尝试更复杂的模型（随机森林、LightGBM）")

# 显示图表
plt.show()

