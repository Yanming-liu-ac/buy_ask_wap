"""
处理订单薄snapshot数据
1. 读取parquet文件
2. 提取从10:00开始每3秒的第一个snapshot
3. 计算ask1_p - bid1_p（价差）
4. 保存处理后的数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, time


def load_orderbook_data(filename='orderbook_snapshots_20251220.parquet'):
    """
    从parquet文件读取订单薄数据
    
    参数:
        filename: parquet文件路径
    
    返回:
        DataFrame
    """
    print(f"读取文件: {filename}")
    df = pd.read_parquet(filename)
    
    # 确保timestamp列是datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"数据读取完成，共 {len(df)} 行")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"列名: {list(df.columns)}")
    
    return df


def extract_sampling_data(df, start_time='10:00:00', interval_seconds=3):
    """
    提取从指定时间开始，每隔指定秒数的第一个snapshot
    
    参数:
        df: 原始订单薄DataFrame
        start_time: 开始时间，格式'HH:MM:SS'
        interval_seconds: 采样间隔（秒）
    
    返回:
        采样后的DataFrame
    """
    print(f"\n开始数据采样...")
    print(f"采样起始时间: {start_time}")
    print(f"采样间隔: {interval_seconds}秒")
    
    # 筛选10:00之后的数据
    df['time'] = df['timestamp'].dt.time
    start_t = datetime.strptime(start_time, '%H:%M:%S').time()
    df_filtered = df[df['time'] >= start_t].copy()
    
    print(f"筛选后数据量: {len(df_filtered)} 行")
    
    # 创建时间分组标签 - 使用floor向下取整
    # 计算从当天0点开始的秒数，然后除以间隔秒数并向下取整
    df_filtered['seconds_from_start'] = (
        df_filtered['timestamp'].dt.hour * 3600 + 
        df_filtered['timestamp'].dt.minute * 60 + 
        df_filtered['timestamp'].dt.second +
        df_filtered['timestamp'].dt.microsecond / 1_000_000
    )
    
    # 计算从10:00:00开始的秒数
    start_seconds = 10 * 3600  # 10:00:00
    df_filtered['seconds_from_10am'] = df_filtered['seconds_from_start'] - start_seconds
    
    # 创建间隔分组（每3秒一组）
    df_filtered['interval_group'] = (df_filtered['seconds_from_10am'] // interval_seconds).astype(int)
    
    # 每组取第一个snapshot
    df_sampled = df_filtered.groupby('interval_group', as_index=False).first()
    
    # 删除辅助列
    df_sampled = df_sampled.drop(['time', 'seconds_from_start', 'seconds_from_10am', 'interval_group'], axis=1)
    
    print(f"采样后数据量: {len(df_sampled)} 行")
    print(f"采样率: {len(df_sampled) / len(df_filtered) * 100:.2f}%")
    
    return df_sampled


def calculate_spread(df):
    """
    计算买卖价差 (ask1_p - bid1_p)
    
    参数:
        df: 订单薄DataFrame
    
    返回:
        添加了spread列的DataFrame
    """
    print("\n计算买卖价差...")
    
    # 计算价差
    df['spread'] = df['ask1_p'] - df['bid1_p']
    
    # 计算中间价
    df['mid_price'] = (df['ask1_p'] + df['bid1_p']) / 2
    
    # 计算相对价差（基点，bp）
    df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
    
    print(f"价差统计信息:")
    print(df['spread'].describe())
    print(f"\n相对价差（基点）统计:")
    print(df['spread_bps'].describe())
    
    return df


def save_processed_data(df, filename='orderbook_sampled.parquet'):
    """
    保存处理后的数据
    
    参数:
        df: 处理后的DataFrame
        filename: 输出文件名
    """
    print(f"\n保存处理后的数据到: {filename}")
    df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
    
    import os
    file_size = os.path.getsize(filename) / 1024  # KB
    print(f"文件保存成功！文件大小: {file_size:.2f} KB")


def main():
    """主流程"""
    # 1. 读取数据
    df = load_orderbook_data('orderbook_snapshots_20251220.parquet')
    
    # 2. 提取从10:00开始每3秒的第一个snapshot
    df_sampled = extract_sampling_data(df, start_time='10:00:00', interval_seconds=3)
    
    # 3. 计算价差
    df_processed = calculate_spread(df_sampled)
    
    # 4. 显示结果样例
    print("\n处理后的数据样例:")
    print(df_processed[['timestamp', 'symbol', 'bid1_p', 'ask1_p', 'spread', 
                         'mid_price', 'spread_bps', 'amount']].head(10))
    
    # 5. 保存处理后的数据
    save_processed_data(df_processed, 'orderbook_sampled.parquet')
    
    print("\n✓ 数据处理完成！")
    
    return df_processed


if __name__ == '__main__':
    df_processed = main()

