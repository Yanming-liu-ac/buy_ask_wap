"""
测试订单和交易特征提取功能

这个脚本用于测试新添加的订单/交易特征提取功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("测试订单/交易特征提取")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1/3] 加载数据...")

# 加载订单薄数据
df = pd.read_parquet('orderbook_sampled_10min_intervals.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"  订单薄数据: {len(df):,} 条记录")

# 加载MBO数据
try:
    mbo = pd.read_csv('mbo.csv')
    mbo['ts_event'] = pd.to_datetime(mbo['ts_event'])
    # 去除时区信息
    if mbo['ts_event'].dt.tz is not None:
        mbo['ts_event'] = mbo['ts_event'].dt.tz_localize(None)
    mbo = mbo.sort_values('ts_event').reset_index(drop=True)
    print(f"  MBO数据: {len(mbo):,} 条记录")
    print(f"    - 时间范围: {mbo['ts_event'].min()} ~ {mbo['ts_event'].max()}")
    if 'side' in mbo.columns:
        print(f"    - 买单: {(mbo['side'] == 'B').sum():,} 条")
        print(f"    - 卖单: {(mbo['side'] == 'S').sum():,} 条")
except FileNotFoundError:
    print("  [警告] mbo.csv 未找到")
    mbo = None

# 加载MBP数据
try:
    mbp = pd.read_csv('mbp.csv')
    mbp['ts_event'] = pd.to_datetime(mbp['ts_event'])
    # 去除时区信息
    if mbp['ts_event'].dt.tz is not None:
        mbp['ts_event'] = mbp['ts_event'].dt.tz_localize(None)
    mbp = mbp.sort_values('ts_event').reset_index(drop=True)
    print(f"  MBP数据: {len(mbp):,} 条记录")
    print(f"    - 时间范围: {mbp['ts_event'].min()} ~ {mbp['ts_event'].max()}")
    if 'side' in mbp.columns:
        print(f"    - 买方成交: {(mbp['side'] == 'B').sum():,} 条")
        print(f"    - 卖方成交: {(mbp['side'] == 'S').sum():,} 条")
except FileNotFoundError:
    print("  [警告] mbp.csv 未找到")
    mbp = None

# ============================================================================
# 2. 测试特征提取（只用前1000条数据测试）
# ============================================================================
print("\n[2/3] 测试特征提取（使用前1000条数据）...")

# 只取前1000条进行测试
df_test = df.head(1000).copy()

# 导入特征提取函数
import sys
sys.path.insert(0, '.')

# 手动定义函数（从price_prediction_linear.py复制）
def create_order_trade_features_test(df, mbo=None, mbp=None, lookback_seconds=5):
    """测试版本的特征提取函数"""
    df = df.copy()
    lookback = pd.Timedelta(seconds=lookback_seconds)
    
    print(f"  - 计算过去{lookback_seconds}秒的订单/交易特征...")
    
    # 处理MBO数据
    if mbo is not None and len(mbo) > 0:
        print("    * 处理订单数据 (MBO)...")
        
        # 过滤时间范围
        min_time = df['timestamp'].min() - lookback
        max_time = df['timestamp'].max()
        mbo_filtered = mbo[(mbo['ts_event'] >= min_time) & (mbo['ts_event'] <= max_time)].copy()
        print(f"      过滤后MBO数据: {len(mbo_filtered):,} 条")
        
        if len(mbo_filtered) > 0 and 'side' in mbo_filtered.columns:
            # 分离买卖
            mbo_buy = mbo_filtered[mbo_filtered['side'] == 'B'].copy()
            mbo_sell = mbo_filtered[mbo_filtered['side'] == 'S'].copy()
            
            # 计算订单金额
            mbo_buy['order_amount'] = mbo_buy['price'] * mbo_buy['size']
            mbo_sell['order_amount'] = mbo_sell['price'] * mbo_sell['size']
            
            # 聚合
            mbo_buy_agg = mbo_buy.groupby('ts_event').agg({
                'size': 'sum',
                'order_amount': 'sum',
                'order_id': 'count'
            }).reset_index()
            mbo_buy_agg.columns = ['ts_event', 'buy_volume', 'buy_amount', 'buy_count']
            mbo_buy_agg = mbo_buy_agg.sort_values('ts_event')
            mbo_buy_agg['buy_volume_cumsum'] = mbo_buy_agg['buy_volume'].cumsum()
            mbo_buy_agg['buy_amount_cumsum'] = mbo_buy_agg['buy_amount'].cumsum()
            mbo_buy_agg['buy_count_cumsum'] = mbo_buy_agg['buy_count'].cumsum()
            
            mbo_sell_agg = mbo_sell.groupby('ts_event').agg({
                'size': 'sum',
                'order_amount': 'sum',
                'order_id': 'count'
            }).reset_index()
            mbo_sell_agg.columns = ['ts_event', 'sell_volume', 'sell_amount', 'sell_count']
            mbo_sell_agg = mbo_sell_agg.sort_values('ts_event')
            mbo_sell_agg['sell_volume_cumsum'] = mbo_sell_agg['sell_volume'].cumsum()
            mbo_sell_agg['sell_amount_cumsum'] = mbo_sell_agg['sell_amount'].cumsum()
            mbo_sell_agg['sell_count_cumsum'] = mbo_sell_agg['sell_count'].cumsum()
            
            # merge
            df['timestamp_start'] = df['timestamp'] - lookback
            
            df = pd.merge_asof(df, mbo_buy_agg[['ts_event', 'buy_volume_cumsum', 'buy_amount_cumsum', 'buy_count_cumsum']],
                              left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
            df = pd.merge_asof(df, mbo_buy_agg[['ts_event', 'buy_volume_cumsum', 'buy_amount_cumsum', 'buy_count_cumsum']],
                              left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
            
            df[f'buy_order_volume_{lookback_seconds}s'] = (df['buy_volume_cumsum_t'] - df['buy_volume_cumsum_t0']).fillna(0)
            df[f'buy_order_amount_{lookback_seconds}s'] = (df['buy_amount_cumsum_t'] - df['buy_amount_cumsum_t0']).fillna(0)
            df[f'buy_order_count_{lookback_seconds}s'] = (df['buy_count_cumsum_t'] - df['buy_count_cumsum_t0']).fillna(0)
            
            df = pd.merge_asof(df, mbo_sell_agg[['ts_event', 'sell_volume_cumsum', 'sell_amount_cumsum', 'sell_count_cumsum']],
                              left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
            df = pd.merge_asof(df, mbo_sell_agg[['ts_event', 'sell_volume_cumsum', 'sell_amount_cumsum', 'sell_count_cumsum']],
                              left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
            
            df[f'sell_order_volume_{lookback_seconds}s'] = (df['sell_volume_cumsum_t'] - df['sell_volume_cumsum_t0']).fillna(0)
            df[f'sell_order_amount_{lookback_seconds}s'] = (df['sell_amount_cumsum_t'] - df['sell_amount_cumsum_t0']).fillna(0)
            df[f'sell_order_count_{lookback_seconds}s'] = (df['sell_count_cumsum_t'] - df['sell_count_cumsum_t0']).fillna(0)
            
            # 清理临时列
            temp_cols = [c for c in df.columns if '_cumsum' in c or c == 'ts_event']
            df = df.drop(columns=temp_cols, errors='ignore')
            
            # 计算不平衡
            df[f'order_flow_imbalance_{lookback_seconds}s'] = (
                (df[f'buy_order_volume_{lookback_seconds}s'] - df[f'sell_order_volume_{lookback_seconds}s']) /
                (df[f'buy_order_volume_{lookback_seconds}s'] + df[f'sell_order_volume_{lookback_seconds}s'] + 1e-8)
            )
            
            print(f"      ✓ 买单数均值={df[f'buy_order_count_{lookback_seconds}s'].mean():.1f}, "
                  f"卖单数均值={df[f'sell_order_count_{lookback_seconds}s'].mean():.1f}")
    
    # 处理MBP数据
    if mbp is not None and len(mbp) > 0:
        print("    * 处理交易数据 (MBP)...")
        
        # 过滤时间范围
        min_time = df['timestamp'].min() - lookback
        max_time = df['timestamp'].max()
        mbp_filtered = mbp[(mbp['ts_event'] >= min_time) & (mbp['ts_event'] <= max_time)].copy()
        print(f"      过滤后MBP数据: {len(mbp_filtered):,} 条")
        
        if len(mbp_filtered) > 0 and 'price' in mbp_filtered.columns and 'size' in mbp_filtered.columns:
            # 计算成交金额
            mbp_filtered['trade_amount'] = mbp_filtered['price'] * mbp_filtered['size']
            
            # 分离买卖
            if 'side' in mbp_filtered.columns:
                trades_buy = mbp_filtered[mbp_filtered['side'] == 'B'].copy()
                trades_sell = mbp_filtered[mbp_filtered['side'] == 'S'].copy()
            else:
                trades_buy = mbp_filtered.copy()
                trades_sell = pd.DataFrame()
            
            # 处理买方成交
            if len(trades_buy) > 0:
                trades_buy_agg = trades_buy.groupby('ts_event').agg({
                    'size': 'sum',
                    'trade_amount': 'sum',
                    'sequence': 'count'
                }).reset_index()
                trades_buy_agg.columns = ['ts_event', 'buy_trade_volume', 'buy_trade_amount', 'buy_trade_count']
                trades_buy_agg = trades_buy_agg.sort_values('ts_event')
                trades_buy_agg['buy_trade_volume_cumsum'] = trades_buy_agg['buy_trade_volume'].cumsum()
                trades_buy_agg['buy_trade_amount_cumsum'] = trades_buy_agg['buy_trade_amount'].cumsum()
                trades_buy_agg['buy_trade_count_cumsum'] = trades_buy_agg['buy_trade_count'].cumsum()
                
                if 'timestamp_start' not in df.columns:
                    df['timestamp_start'] = df['timestamp'] - lookback
                
                df = pd.merge_asof(df, trades_buy_agg[['ts_event', 'buy_trade_volume_cumsum', 'buy_trade_amount_cumsum', 'buy_trade_count_cumsum']],
                                  left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
                df = pd.merge_asof(df, trades_buy_agg[['ts_event', 'buy_trade_volume_cumsum', 'buy_trade_amount_cumsum', 'buy_trade_count_cumsum']],
                                  left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
                
                df[f'buy_trade_volume_{lookback_seconds}s'] = (df['buy_trade_volume_cumsum_t'] - df['buy_trade_volume_cumsum_t0']).fillna(0)
                df[f'buy_trade_amount_{lookback_seconds}s'] = (df['buy_trade_amount_cumsum_t'] - df['buy_trade_amount_cumsum_t0']).fillna(0)
                df[f'buy_trade_count_{lookback_seconds}s'] = (df['buy_trade_count_cumsum_t'] - df['buy_trade_count_cumsum_t0']).fillna(0)
            
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
                
                df = pd.merge_asof(df, trades_sell_agg[['ts_event', 'sell_trade_volume_cumsum', 'sell_trade_amount_cumsum', 'sell_trade_count_cumsum']],
                                  left_on='timestamp', right_on='ts_event', direction='backward', suffixes=('', '_t'))
                df = pd.merge_asof(df, trades_sell_agg[['ts_event', 'sell_trade_volume_cumsum', 'sell_trade_amount_cumsum', 'sell_trade_count_cumsum']],
                                  left_on='timestamp_start', right_on='ts_event', direction='backward', suffixes=('_t', '_t0'))
                
                df[f'sell_trade_volume_{lookback_seconds}s'] = (df['sell_trade_volume_cumsum_t'] - df['sell_trade_volume_cumsum_t0']).fillna(0)
                df[f'sell_trade_amount_{lookback_seconds}s'] = (df['sell_trade_amount_cumsum_t'] - df['sell_trade_amount_cumsum_t0']).fillna(0)
                df[f'sell_trade_count_{lookback_seconds}s'] = (df['sell_trade_count_cumsum_t'] - df['sell_trade_count_cumsum_t0']).fillna(0)
            
            # 清理临时列
            temp_cols = [c for c in df.columns if '_cumsum' in c or c == 'ts_event']
            df = df.drop(columns=temp_cols, errors='ignore')
            
            # 计算不平衡和强度
            if f'buy_trade_volume_{lookback_seconds}s' in df.columns and f'sell_trade_volume_{lookback_seconds}s' in df.columns:
                df[f'trade_volume_imbalance_{lookback_seconds}s'] = (
                    (df[f'buy_trade_volume_{lookback_seconds}s'] - df[f'sell_trade_volume_{lookback_seconds}s']) /
                    (df[f'buy_trade_volume_{lookback_seconds}s'] + df[f'sell_trade_volume_{lookback_seconds}s'] + 1e-8)
                )
                
                df[f'trade_intensity_{lookback_seconds}s'] = (
                    (df[f'buy_trade_volume_{lookback_seconds}s'] + df[f'sell_trade_volume_{lookback_seconds}s']) / 
                    lookback_seconds
                )
                
                print(f"      ✓ 买方成交均值={df[f'buy_trade_count_{lookback_seconds}s'].mean():.1f}, "
                      f"卖方成交均值={df[f'sell_trade_count_{lookback_seconds}s'].mean():.1f}")
    
    # 清理
    if 'timestamp_start' in df.columns:
        df = df.drop(columns=['timestamp_start'])
    
    return df

# 执行测试
df_result = create_order_trade_features_test(df_test, mbo, mbp, lookback_seconds=5)

# ============================================================================
# 3. 检查结果
# ============================================================================
print("\n[3/3] 检查结果...")

# 找出新增的特征列
new_cols = [c for c in df_result.columns if c not in df_test.columns]
print(f"\n新增特征数量: {len(new_cols)}")
print(f"新增特征列表:")
for col in new_cols:
    non_zero = (df_result[col] != 0).sum()
    mean_val = df_result[col].mean()
    max_val = df_result[col].max()
    print(f"  - {col:40s}: 非零={non_zero:4d}, 均值={mean_val:10.2f}, 最大值={max_val:10.2f}")

# 显示示例数据
print("\n示例数据（前5行）:")
display_cols = ['timestamp'] + new_cols[:5]
print(df_result[display_cols].head())

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

