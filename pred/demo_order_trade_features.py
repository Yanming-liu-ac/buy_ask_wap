"""
订单和交易特征提取功能演示

由于当前MBO/MBP数据与订单薄数据时间不匹配，
这个脚本创建模拟数据来演示功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("订单/交易特征提取功能演示")
print("=" * 80)

# ============================================================================
# 1. 创建模拟数据
# ============================================================================
print("\n[1/4] 创建模拟数据...")

# 创建订单薄数据（每3秒一个snapshot）
start_time = pd.Timestamp('2025-12-20 10:00:00')
timestamps = pd.date_range(start_time, periods=100, freq='3s')

df = pd.DataFrame({
    'timestamp': timestamps,
    'mid_price': 100 + np.cumsum(np.random.randn(100) * 0.1),
    'spread': np.random.uniform(0.01, 0.05, 100),
    'bid1_v': np.random.randint(1000, 10000, 100),
    'ask1_v': np.random.randint(1000, 10000, 100),
})

print(f"  订单薄数据: {len(df)} 条记录")
print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# 创建MBO数据（订单数据）
# 在每个3秒区间内随机生成一些订单
mbo_records = []
for i in range(len(timestamps) - 1):
    t_start = timestamps[i]
    t_end = timestamps[i + 1]
    
    # 每个区间生成5-15个订单
    n_orders = np.random.randint(5, 15)
    
    for j in range(n_orders):
        # 随机时间
        t = t_start + pd.Timedelta(seconds=np.random.uniform(0, 3))
        
        # 随机买卖方向
        side = np.random.choice(['B', 'S'])
        
        # 随机价格和数量
        base_price = df.loc[i, 'mid_price']
        price = base_price + np.random.uniform(-0.5, 0.5)
        size = np.random.randint(100, 1000)
        
        mbo_records.append({
            'ts_event': t,
            'side': side,
            'price': price,
            'size': size,
            'order_id': len(mbo_records) + 1
        })

mbo = pd.DataFrame(mbo_records)
mbo = mbo.sort_values('ts_event').reset_index(drop=True)

print(f"  MBO数据: {len(mbo)} 条记录")
print(f"    - 买单: {(mbo['side'] == 'B').sum()} 条")
print(f"    - 卖单: {(mbo['side'] == 'S').sum()} 条")

# 创建MBP数据（成交数据）
# 基于MBO生成一些成交
mbp_records = []
for i in range(len(mbo) // 3):  # 假设1/3的订单成交
    order = mbo.iloc[i * 3]
    
    mbp_records.append({
        'ts_event': order['ts_event'] + pd.Timedelta(milliseconds=np.random.randint(10, 100)),
        'side': order['side'],
        'price': order['price'],
        'size': order['size'] // 2,  # 部分成交
        'sequence': len(mbp_records) + 1
    })

mbp = pd.DataFrame(mbp_records)
mbp = mbp.sort_values('ts_event').reset_index(drop=True)

print(f"  MBP数据: {len(mbp)} 条记录")
print(f"    - 买方成交: {(mbp['side'] == 'B').sum()} 条")
print(f"    - 卖方成交: {(mbp['side'] == 'S').sum()} 条")

# ============================================================================
# 2. 提取特征
# ============================================================================
print("\n[2/4] 提取订单/交易特征...")

def create_order_trade_features(df, mbo=None, mbp=None, lookback_seconds=5):
    """提取订单和交易特征（矢量化版本）"""
    df = df.copy()
    lookback = pd.Timedelta(seconds=lookback_seconds)
    
    # 处理MBO数据
    if mbo is not None and len(mbo) > 0:
        print("  - 处理订单数据...")
        
        # 分离买卖
        mbo_buy = mbo[mbo['side'] == 'B'].copy()
        mbo_sell = mbo[mbo['side'] == 'S'].copy()
        
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
                          left_on='timestamp', right_on='ts_event', direction='backward')
        df = df.rename(columns={'buy_volume_cumsum': 'buy_volume_cumsum_t', 
                                'buy_amount_cumsum': 'buy_amount_cumsum_t',
                                'buy_count_cumsum': 'buy_count_cumsum_t'})
        df = df.drop(columns=['ts_event'], errors='ignore')
        
        df = pd.merge_asof(df, mbo_buy_agg[['ts_event', 'buy_volume_cumsum', 'buy_amount_cumsum', 'buy_count_cumsum']],
                          left_on='timestamp_start', right_on='ts_event', direction='backward')
        df = df.rename(columns={'buy_volume_cumsum': 'buy_volume_cumsum_t0', 
                                'buy_amount_cumsum': 'buy_amount_cumsum_t0',
                                'buy_count_cumsum': 'buy_count_cumsum_t0'})
        df = df.drop(columns=['ts_event'], errors='ignore')
        
        df[f'buy_order_volume_{lookback_seconds}s'] = (df['buy_volume_cumsum_t'] - df['buy_volume_cumsum_t0']).fillna(0)
        df[f'buy_order_amount_{lookback_seconds}s'] = (df['buy_amount_cumsum_t'] - df['buy_amount_cumsum_t0']).fillna(0)
        df[f'buy_order_count_{lookback_seconds}s'] = (df['buy_count_cumsum_t'] - df['buy_count_cumsum_t0']).fillna(0)
        
        df = pd.merge_asof(df, mbo_sell_agg[['ts_event', 'sell_volume_cumsum', 'sell_amount_cumsum', 'sell_count_cumsum']],
                          left_on='timestamp', right_on='ts_event', direction='backward')
        df = df.rename(columns={'sell_volume_cumsum': 'sell_volume_cumsum_t', 
                                'sell_amount_cumsum': 'sell_amount_cumsum_t',
                                'sell_count_cumsum': 'sell_count_cumsum_t'})
        df = df.drop(columns=['ts_event'], errors='ignore')
        
        df = pd.merge_asof(df, mbo_sell_agg[['ts_event', 'sell_volume_cumsum', 'sell_amount_cumsum', 'sell_count_cumsum']],
                          left_on='timestamp_start', right_on='ts_event', direction='backward')
        df = df.rename(columns={'sell_volume_cumsum': 'sell_volume_cumsum_t0', 
                                'sell_amount_cumsum': 'sell_amount_cumsum_t0',
                                'sell_count_cumsum': 'sell_count_cumsum_t0'})
        df = df.drop(columns=['ts_event'], errors='ignore')
        
        df[f'sell_order_volume_{lookback_seconds}s'] = (df['sell_volume_cumsum_t'] - df['sell_volume_cumsum_t0']).fillna(0)
        df[f'sell_order_amount_{lookback_seconds}s'] = (df['sell_amount_cumsum_t'] - df['sell_amount_cumsum_t0']).fillna(0)
        df[f'sell_order_count_{lookback_seconds}s'] = (df['sell_count_cumsum_t'] - df['sell_count_cumsum_t0']).fillna(0)
        
        # 清理所有临时列
        temp_cols = [c for c in df.columns if ('_cumsum' in c or 'ts_event' in c)]
        df = df.drop(columns=temp_cols, errors='ignore')
        
        # 计算不平衡
        df[f'order_flow_imbalance_{lookback_seconds}s'] = (
            (df[f'buy_order_volume_{lookback_seconds}s'] - df[f'sell_order_volume_{lookback_seconds}s']) /
            (df[f'buy_order_volume_{lookback_seconds}s'] + df[f'sell_order_volume_{lookback_seconds}s'] + 1e-8)
        )
        
        df[f'order_amount_imbalance_{lookback_seconds}s'] = (
            (df[f'buy_order_amount_{lookback_seconds}s'] - df[f'sell_order_amount_{lookback_seconds}s']) /
            (df[f'buy_order_amount_{lookback_seconds}s'] + df[f'sell_order_amount_{lookback_seconds}s'] + 1e-8)
        )
        
        print(f"    [OK] 买单数均值={df[f'buy_order_count_{lookback_seconds}s'].mean():.1f}, "
              f"卖单数均值={df[f'sell_order_count_{lookback_seconds}s'].mean():.1f}")
    
    # 处理MBP数据
    if mbp is not None and len(mbp) > 0:
        print("  - 处理交易数据...")
        
        mbp['trade_amount'] = mbp['price'] * mbp['size']
        
        trades_buy = mbp[mbp['side'] == 'B'].copy()
        trades_sell = mbp[mbp['side'] == 'S'].copy()
        
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
                              left_on='timestamp', right_on='ts_event', direction='backward')
            df = df.rename(columns={'buy_trade_volume_cumsum': 'buy_trade_volume_cumsum_t', 
                                    'buy_trade_amount_cumsum': 'buy_trade_amount_cumsum_t',
                                    'buy_trade_count_cumsum': 'buy_trade_count_cumsum_t'})
            df = df.drop(columns=['ts_event'], errors='ignore')
            
            df = pd.merge_asof(df, trades_buy_agg[['ts_event', 'buy_trade_volume_cumsum', 'buy_trade_amount_cumsum', 'buy_trade_count_cumsum']],
                              left_on='timestamp_start', right_on='ts_event', direction='backward')
            df = df.rename(columns={'buy_trade_volume_cumsum': 'buy_trade_volume_cumsum_t0', 
                                    'buy_trade_amount_cumsum': 'buy_trade_amount_cumsum_t0',
                                    'buy_trade_count_cumsum': 'buy_trade_count_cumsum_t0'})
            df = df.drop(columns=['ts_event'], errors='ignore')
            
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
                              left_on='timestamp', right_on='ts_event', direction='backward')
            df = df.rename(columns={'sell_trade_volume_cumsum': 'sell_trade_volume_cumsum_t', 
                                    'sell_trade_amount_cumsum': 'sell_trade_amount_cumsum_t',
                                    'sell_trade_count_cumsum': 'sell_trade_count_cumsum_t'})
            df = df.drop(columns=['ts_event'], errors='ignore')
            
            df = pd.merge_asof(df, trades_sell_agg[['ts_event', 'sell_trade_volume_cumsum', 'sell_trade_amount_cumsum', 'sell_trade_count_cumsum']],
                              left_on='timestamp_start', right_on='ts_event', direction='backward')
            df = df.rename(columns={'sell_trade_volume_cumsum': 'sell_trade_volume_cumsum_t0', 
                                    'sell_trade_amount_cumsum': 'sell_trade_amount_cumsum_t0',
                                    'sell_trade_count_cumsum': 'sell_trade_count_cumsum_t0'})
            df = df.drop(columns=['ts_event'], errors='ignore')
            
            df[f'sell_trade_volume_{lookback_seconds}s'] = (df['sell_trade_volume_cumsum_t'] - df['sell_trade_volume_cumsum_t0']).fillna(0)
            df[f'sell_trade_amount_{lookback_seconds}s'] = (df['sell_trade_amount_cumsum_t'] - df['sell_trade_amount_cumsum_t0']).fillna(0)
            df[f'sell_trade_count_{lookback_seconds}s'] = (df['sell_trade_count_cumsum_t'] - df['sell_trade_count_cumsum_t0']).fillna(0)
            
            # 清理所有临时列
            temp_cols = [c for c in df.columns if '_cumsum' in c]
            df = df.drop(columns=temp_cols, errors='ignore')
        
        # 计算不平衡和强度
        if f'buy_trade_volume_{lookback_seconds}s' in df.columns and f'sell_trade_volume_{lookback_seconds}s' in df.columns:
            df[f'trade_volume_imbalance_{lookback_seconds}s'] = (
                (df[f'buy_trade_volume_{lookback_seconds}s'] - df[f'sell_trade_volume_{lookback_seconds}s']) /
                (df[f'buy_trade_volume_{lookback_seconds}s'] + df[f'sell_trade_volume_{lookback_seconds}s'] + 1e-8)
            )
            
            df[f'trade_amount_imbalance_{lookback_seconds}s'] = (
                (df[f'buy_trade_amount_{lookback_seconds}s'] - df[f'sell_trade_amount_{lookback_seconds}s']) /
                (df[f'buy_trade_amount_{lookback_seconds}s'] + df[f'sell_trade_amount_{lookback_seconds}s'] + 1e-8)
            )
            
            df[f'trade_intensity_{lookback_seconds}s'] = (
                (df[f'buy_trade_volume_{lookback_seconds}s'] + df[f'sell_trade_volume_{lookback_seconds}s']) / 
                lookback_seconds
            )
            
            print(f"    [OK] 买方成交均值={df[f'buy_trade_count_{lookback_seconds}s'].mean():.1f}, "
                  f"卖方成交均值={df[f'sell_trade_count_{lookback_seconds}s'].mean():.1f}")
    
    # 清理
    if 'timestamp_start' in df.columns:
        df = df.drop(columns=['timestamp_start'])
    
    return df

# 执行特征提取
df_result = create_order_trade_features(df, mbo, mbp, lookback_seconds=5)

# ============================================================================
# 3. 展示结果
# ============================================================================
print("\n[3/4] 特征提取结果...")

# 找出新增的特征
new_cols = [c for c in df_result.columns if c not in df.columns]
print(f"\n新增特征数量: {len(new_cols)}")
print(f"\n新增特征统计:")
for col in new_cols:
    non_zero = (df_result[col] != 0).sum()
    mean_val = df_result[col].mean()
    std_val = df_result[col].std()
    min_val = df_result[col].min()
    max_val = df_result[col].max()
    print(f"  {col:45s}: 非零={non_zero:3d}, 均值={mean_val:10.2f}, 标准差={std_val:10.2f}, 范围=[{min_val:8.2f}, {max_val:8.2f}]")

# ============================================================================
# 4. 可视化
# ============================================================================
print("\n[4/4] 生成可视化...")

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# 子图1: 订单数量
ax = axes[0, 0]
ax.plot(df_result['timestamp'], df_result['buy_order_count_5s'], label='买单数', alpha=0.7)
ax.plot(df_result['timestamp'], df_result['sell_order_count_5s'], label='卖单数', alpha=0.7)
ax.set_title('过去5秒订单数量', fontweight='bold')
ax.set_ylabel('订单数')
ax.legend()
ax.grid(alpha=0.3)

# 子图2: 订单金额
ax = axes[0, 1]
ax.plot(df_result['timestamp'], df_result['buy_order_amount_5s'], label='买单金额', alpha=0.7)
ax.plot(df_result['timestamp'], df_result['sell_order_amount_5s'], label='卖单金额', alpha=0.7)
ax.set_title('过去5秒订单金额', fontweight='bold')
ax.set_ylabel('金额（元）')
ax.legend()
ax.grid(alpha=0.3)

# 子图3: 订单流不平衡
ax = axes[1, 0]
ax.plot(df_result['timestamp'], df_result['order_flow_imbalance_5s'], color='purple', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.fill_between(df_result['timestamp'], df_result['order_flow_imbalance_5s'], 0, 
                alpha=0.3, color='purple')
ax.set_title('订单流不平衡度', fontweight='bold')
ax.set_ylabel('不平衡度')
ax.grid(alpha=0.3)

# 子图4: 成交数量
ax = axes[1, 1]
ax.plot(df_result['timestamp'], df_result['buy_trade_count_5s'], label='买方成交', alpha=0.7)
ax.plot(df_result['timestamp'], df_result['sell_trade_count_5s'], label='卖方成交', alpha=0.7)
ax.set_title('过去5秒成交笔数', fontweight='bold')
ax.set_ylabel('成交笔数')
ax.legend()
ax.grid(alpha=0.3)

# 子图5: 成交不平衡
ax = axes[2, 0]
ax.plot(df_result['timestamp'], df_result['trade_volume_imbalance_5s'], color='green', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.fill_between(df_result['timestamp'], df_result['trade_volume_imbalance_5s'], 0, 
                alpha=0.3, color='green')
ax.set_title('成交不平衡度', fontweight='bold')
ax.set_ylabel('不平衡度')
ax.grid(alpha=0.3)

# 子图6: 交易强度
ax = axes[2, 1]
ax.plot(df_result['timestamp'], df_result['trade_intensity_5s'], color='red', alpha=0.7)
ax.set_title('交易强度', fontweight='bold')
ax.set_ylabel('成交量/秒')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('order_trade_features_demo.png', dpi=150, bbox_inches='tight')
print("  [OK] 保存图表: order_trade_features_demo.png")

print("\n" + "=" * 80)
print("演示完成！")
print("=" * 80)
print("\n说明:")
print("1. 这是使用模拟数据的演示")
print("2. 实际使用时需要真实的MBO和MBP数据")
print("3. 数据的时间范围必须匹配")
print("4. 详细文档请查看: 订单交易特征说明.md")

