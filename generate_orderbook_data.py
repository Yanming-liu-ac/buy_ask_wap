"""
生成模拟的订单薄snapshot数据
模拟一整天的交易数据（9:30 - 15:00）

数据字段说明：
- volume: 累计成交量（整数，股数）
- amount: 累计成交额（浮点数，金额）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa


def generate_orderbook_snapshots(date_str='2025-12-20'):
    """
    生成一整天的订单薄snapshot数据
    
    参数:
        date_str: 日期字符串，格式为'YYYY-MM-DD'
    
    返回:
        DataFrame包含订单薄数据
    """
    print(f"开始生成 {date_str} 的订单薄数据...")
    
    # 设置交易时间段: 9:30 - 15:00
    start_time = datetime.strptime(f"{date_str} 09:30:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(f"{date_str} 15:00:00", "%Y-%m-%d %H:%M:%S")
    
    # 生成时间序列，每100毫秒一个snapshot
    time_range = pd.date_range(start=start_time, end=end_time, freq='100ms')
    
    print(f"总共生成 {len(time_range)} 个snapshot数据点")
    
    # 初始化价格和数量
    base_price = 100.0  # 基准价格
    np.random.seed(42)
    
    data = []
    
    for idx, timestamp in enumerate(time_range):
        # 模拟价格波动
        price_drift = np.sin(idx / 10000) * 5  # 趋势性波动
        noise = np.random.randn() * 0.1  # 随机噪声
        mid_price = base_price + price_drift + noise
        
        # 买卖价差
        spread = 0.01 + abs(np.random.randn() * 0.01)
        
        # 买一价和卖一价
        bid1_p = mid_price - spread / 2
        ask1_p = mid_price + spread / 2
        
        # 买一量和卖一量
        bid1_v = np.random.randint(100, 10000)
        ask1_v = np.random.randint(100, 10000)
        
        # 五档行情数据
        bid_prices = [bid1_p - i * 0.01 for i in range(5)]
        ask_prices = [ask1_p + i * 0.01 for i in range(5)]
        bid_volumes = [np.random.randint(100, 10000) for _ in range(5)]
        ask_volumes = [np.random.randint(100, 10000) for _ in range(5)]
        
        # 累计成交量（整数）- 随着时间推移逐渐增加
        # 每个snapshot新增一些交易量
        new_trades = np.random.poisson(50)  # 泊松分布模拟每个时刻的新交易
        volume = idx * 50 + new_trades  # 累计交易量（整数）
        
        # 最新成交价
        last_price = mid_price + np.random.randn() * 0.005
        
        # 累计成交额 = 累计成交量 * 平均成交价
        amount = volume * mid_price
        
        snapshot = {
            'timestamp': timestamp,
            'symbol': 'TEST001',
            'bid1_p': round(bid1_p, 2),
            'bid1_v': bid1_v,
            'bid2_p': round(bid_prices[1], 2),
            'bid2_v': bid_volumes[1],
            'bid3_p': round(bid_prices[2], 2),
            'bid3_v': bid_volumes[2],
            'bid4_p': round(bid_prices[3], 2),
            'bid4_v': bid_volumes[3],
            'bid5_p': round(bid_prices[4], 2),
            'bid5_v': bid_volumes[4],
            'ask1_p': round(ask1_p, 2),
            'ask1_v': ask1_v,
            'ask2_p': round(ask_prices[1], 2),
            'ask2_v': ask_volumes[1],
            'ask3_p': round(ask_prices[2], 2),
            'ask3_v': ask_volumes[2],
            'ask4_p': round(ask_prices[3], 2),
            'ask4_v': ask_volumes[3],
            'ask5_p': round(ask_prices[4], 2),
            'ask5_v': ask_volumes[4],
            'last_price': round(last_price, 2),
            'volume': int(volume),  # 累计成交量（整数）
            'amount': round(amount, 2),  # 累计成交额（金额）
        }
        
        data.append(snapshot)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    print(f"数据生成完成，共 {len(df)} 行")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 验证数据类型和字段
    print(f"\n字段验证:")
    print(f"  - volume类型: {df['volume'].dtype} (应为整数)")
    print(f"  - amount类型: {df['amount'].dtype} (应为浮点数)")
    print(f"  - volume范围: [{df['volume'].min():,} ~ {df['volume'].max():,}]")
    print(f"  - amount范围: [{df['amount'].min():,.2f} ~ {df['amount'].max():,.2f}]")
    
    # 验证volume是否为整数
    assert df['volume'].dtype in [np.int32, np.int64], "volume应该是整数类型!"
    print(f"  ✓ volume字段验证通过（整数类型）")
    
    print(f"\n数据样例:")
    print(df[['timestamp', 'symbol', 'bid1_p', 'ask1_p', 'last_price', 'volume', 'amount']].head(10))
    
    print(f"\n数据信息:")
    print(df.info())
    
    return df


def save_to_parquet(df, filename='orderbook_snapshots.parquet'):
    """
    保存DataFrame到parquet文件
    
    参数:
        df: 订单薄DataFrame
        filename: 输出文件名
    """
    print(f"\n保存数据到 {filename}...")
    df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
    
    # 显示文件大小
    import os
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"文件保存成功！文件大小: {file_size:.2f} MB")


if __name__ == '__main__':
    # 生成数据
    df = generate_orderbook_snapshots('2025-12-20')
    
    # 额外验证
    print("\n" + "="*60)
    print("数据完整性检查:")
    print("="*60)
    print(f"总记录数: {len(df):,}")
    print(f"字段数量: {len(df.columns)}")
    print(f"\n关键字段说明:")
    print(f"  - volume (累计成交量): {df['volume'].iloc[-1]:,} 股")
    print(f"  - amount (累计成交额): {df['amount'].iloc[-1]:,.2f} 元")
    print(f"  - 最终平均成交价: {df['amount'].iloc[-1] / df['volume'].iloc[-1]:.2f} 元/股")
    
    # 保存为parquet格式
    save_to_parquet(df, 'orderbook_snapshots_20251220.parquet')
    
    print("\n✓ 数据生成完成！")

