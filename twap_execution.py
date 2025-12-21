"""
TWAP交易执行策略
基于订单薄数据实施TWAP策略：
- 每10分钟区间交易该区间amount的1%
- 在10分钟内均匀分配交易量
"""
# 导入必要的库
import pandas as pd  # pandas是Python中用于数据处理的核心库，简称pd
import numpy as np   # numpy是数值计算库，简称np
from datetime import datetime, timedelta  # datetime用于处理日期和时间


def load_sampled_data(filename='orderbook_sampled.parquet'):
    """
    读取采样后的订单薄数据
    
    参数:
        filename: parquet文件路径（默认值是'orderbook_sampled.parquet'）
    
    返回:
        DataFrame（pandas的表格数据结构，类似Excel表格）
    """
    # 打印正在读取的文件名（f-string语法：f"文本{变量}"可以把变量嵌入字符串）
    print(f"读取采样数据: {filename}")
    
    # pd.read_parquet()函数读取parquet格式文件，返回DataFrame
    # df是DataFrame的常用简称
    df = pd.read_parquet(filename)
    
    # 确保timestamp列是datetime类型（时间戳类型）
    # pd.to_datetime()可以把各种格式的时间字符串转换为标准的datetime对象
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # len(df)返回DataFrame的行数
    print(f"数据读取完成，共 {len(df)} 行")
    
    # .min()和.max()分别返回该列的最小值和最大值
    # 对于时间列，就是最早和最晚的时间
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 函数返回读取的DataFrame
    return df


def create_time_intervals(df, interval_minutes=10):
    """
    创建时间区间标签（每10分钟一个区间）
    
    参数:
        df: 订单薄DataFrame（表格数据）
        interval_minutes: 区间长度（分钟，默认10分钟）
    
    返回:
        添加了interval_label列的DataFrame
    """
    print(f"\n创建{interval_minutes}分钟时间区间...")
    
    # .sort_values()按指定列排序，这里按时间排序
    # .copy()创建副本，避免修改原数据（pandas的好习惯）
    df = df.sort_values('timestamp').copy()
    
    # 从timestamp列提取小时和分钟
    # .dt是pandas的datetime访问器，可以访问时间的各个部分
    # 例如：timestamp = "2025-12-20 10:30:45"
    #      timestamp.dt.hour = 10（小时）
    #      timestamp.dt.minute = 30（分钟）
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # 计算从当天0点开始经过了多少分钟
    # 例如：10:30 = 10*60 + 30 = 630分钟
    df['minutes_from_start'] = df['hour'] * 60 + df['minute']
    
    # 用整除运算符 // 创建区间编号
    # 例如：630 // 10 = 63（第63个10分钟区间）
    #      635 // 10 = 63（还是第63个区间）
    #      640 // 10 = 64（第64个区间）
    # 这样同一个10分钟内的所有时间点都有相同的interval_id
    df['interval_id'] = df['minutes_from_start'] // interval_minutes
    
    # 定义一个内部函数，用于格式化区间标签
    # 这个函数会把区间编号转换成易读的时间范围字符串
    def format_interval(row):
        # row是DataFrame的一行数据
        # 计算区间开始时间（单位：分钟）
        start_min = (row['interval_id'] * interval_minutes)
        
        # 把分钟转换回小时和分钟
        # // 是整除，% 是取余数
        # 例如：630 // 60 = 10小时，630 % 60 = 30分钟
        start_hour = start_min // 60
        start_minute = start_min % 60
        
        # 计算区间结束时间
        end_min = start_min + interval_minutes
        end_hour = end_min // 60
        end_minute = end_min % 60
        
        # 格式化为字符串，:02d表示两位数字，不足补0
        # 例如：9变成09，15还是15
        return f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"
    
    # .apply()对每一行应用format_interval函数
    # axis=1表示按行操作（axis=0是按列操作）
    df['interval_label'] = df.apply(format_interval, axis=1)
    
    # .nunique()返回不重复值的数量
    print(f"共创建 {df['interval_id'].nunique()} 个时间区间")
    
    # .unique()返回所有不重复的值
    # [:5]是切片操作，取前5个
    print(f"区间示例: {df['interval_label'].unique()[:5]}")
    
    return df


def calculate_interval_metrics(df):
    """
    计算每个时间区间的统计指标
    
    参数:
        df: 带有interval_id的DataFrame
    
    返回:
        区间统计DataFrame（每个区间一行，包含各种统计指标）
    """
    print("\n计算每个区间的统计指标...")
    
    # groupby()是pandas中最重要的操作之一：分组
    # 按interval_id和interval_label分组，然后对每组进行聚合统计
    # .agg()是聚合函数，可以对不同列应用不同的统计方法
    interval_stats = df.groupby(['interval_id', 'interval_label']).agg({
        # 对timestamp列：计算最小值、最大值、计数
        'timestamp': ['min', 'max', 'count'],  # 起止时间和数据点数
        
        # 对amount列（累积成交量）：取区间开始和结束时的值
        # 'first'是第一个值，'last'是最后一个值
        'amount': ['first', 'last'],  
        
        # 对mid_price列：计算均值、标准差、最小值、最大值
        'mid_price': ['mean', 'std', 'min', 'max'],  
        
        # 对spread列：计算均值和标准差
        'spread': ['mean', 'std'],  
        
        # 对spread_bps列：只计算均值（单个统计可以直接写字符串）
        'spread_bps': 'mean',  
        
        # 对volume列：取首尾值
        'volume': ['first', 'last'],  
    }).reset_index()  # reset_index()把分组的键变回普通列
    
    # 上面的操作会产生多层列名，例如：('timestamp', 'min')
    # 下面的代码将多层列名扁平化为单层：'timestamp_min'
    # 这是一个列表推导式（Python的高级语法）
    # 如果col[1]存在（有第二层），就用'_'连接；否则只用col[0]
    interval_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                               for col in interval_stats.columns]
    
    # 计算区间内的增量成交量
    # 区间结束时的累积量 - 区间开始时的累积量 = 区间内新增量
    # 这就是这个10分钟内实际发生的成交量
    interval_stats['interval_amount'] = (interval_stats['amount_last'] - 
                                          interval_stats['amount_first'])
    
    # 同样计算区间内的增量成交额
    interval_stats['interval_volume'] = (interval_stats['volume_last'] - 
                                          interval_stats['volume_first'])
    
    print(f"\n区间统计样例:")
    # 选择几个关键列打印出来看看
    # [列名列表]是pandas选择多列的语法
    print(interval_stats[['interval_label', 'timestamp_count', 'interval_amount', 
                           'mid_price_mean', 'spread_mean']].head())
    
    return interval_stats


def calculate_twap_execution(interval_stats, execution_ratio=0.01):
    """
    计算TWAP执行策略
    
    参数:
        interval_stats: 区间统计DataFrame
        execution_ratio: 执行比例（默认1%，即0.01）
    
    返回:
        添加了TWAP执行计划的DataFrame
    """
    print(f"\n计算TWAP执行策略（执行比例: {execution_ratio*100}%）...")
    
    # 计算目标执行量（区间amount的1%）
    # 例如：区间成交了100000股，我们要执行其中的1% = 1000股
    # 这里用向量化操作：整列 * 标量 = 整列每个元素都乘以这个数
    interval_stats['target_execution_amount'] = (interval_stats['interval_amount'] * 
                                                   execution_ratio)
    
    # 计算每个数据点（每3秒）应该执行的量
    # TWAP的核心思想：Time-Weighted Average Price（时间加权平均价格）
    # 把总执行量均匀分配到每个时间点
    # 例如：10分钟要执行1000股，有200个数据点，每个点执行 1000/200 = 5股
    interval_stats['execution_per_snapshot'] = (interval_stats['target_execution_amount'] / 
                                                 interval_stats['timestamp_count'])
    
    # 计算预估执行成本（使用平均中间价）
    # 成本 = 执行量 × 价格
    # 这里用区间内的平均中间价来估算
    interval_stats['estimated_cost'] = (interval_stats['target_execution_amount'] * 
                                         interval_stats['mid_price_mean'])
    
    # 计算预估滑点成本
    # 滑点(slippage)：实际成交价与理论价格的差异
    # 假设我们以ask价（卖一价）买入，相比中间价会多付出 spread/2 的成本
    # 滑点成本 = 执行量 × 平均价差 / 2
    interval_stats['estimated_slippage'] = (interval_stats['target_execution_amount'] * 
                                             interval_stats['spread_mean'] / 2)
    
    print(f"\nTWAP执行计划样例:")
    # 显示关键列的前10行
    print(interval_stats[['interval_label', 'interval_amount', 'target_execution_amount',
                           'execution_per_snapshot', 'estimated_cost', 
                           'estimated_slippage']].head(10))
    
    # === 汇总统计 ===
    print(f"\n执行计划汇总:")
    print(f"总区间数: {len(interval_stats)}")
    
    # .sum()对整列求和
    # :,.0f 是格式化：逗号分隔千位，0位小数
    print(f"总目标执行量: {interval_stats['target_execution_amount'].sum():,.0f}")
    
    # :.2f 表示保留2位小数
    print(f"预估总成本: ${interval_stats['estimated_cost'].sum():,.2f}")
    print(f"预估总滑点: ${interval_stats['estimated_slippage'].sum():,.2f}")
    
    # .mean()计算平均值
    print(f"平均每区间执行量: {interval_stats['target_execution_amount'].mean():,.0f}")
    
    return interval_stats


def generate_detailed_execution_schedule(df, interval_stats):
    """
    生成详细的逐笔执行计划
    
    参数:
        df: 原始采样数据（每3秒一个点）
        interval_stats: 区间统计数据
    
    返回:
        详细执行计划DataFrame（每行代表一个时间点的执行计划）
    """
    print("\n生成详细执行计划...")
    
    # merge()是pandas的合并操作，类似SQL的JOIN
    # 把区间统计信息合并到每个snapshot
    # on='interval_id'：按interval_id列匹配
    # how='left'：左连接，保留df中的所有行
    df_execution = df.merge(
        # 从interval_stats中只选择需要的列
        interval_stats[['interval_id', 'target_execution_amount', 
                        'execution_per_snapshot', 'timestamp_count']], 
        on='interval_id',  # 连接键
        how='left'  # 左连接
    )
    
    # 计算每个时间点的执行量
    # 这里就是把每个snapshot应该执行的量赋值给execution_amount列
    df_execution['execution_amount'] = df_execution['execution_per_snapshot']
    
    # 计算执行价格（假设以ask价买入）
    # 买入时要"吃"卖单，所以价格是ask1_p（卖一价）
    df_execution['execution_price'] = df_execution['ask1_p']
    
    # 计算执行成本
    # 成本 = 执行数量 × 执行价格
    # 例如：买100股，每股$50，成本=$5000
    df_execution['execution_cost'] = (df_execution['execution_amount'] * 
                                       df_execution['execution_price'])
    
    # 计算滑点（相对于中间价）
    # 滑点 = 实际成交价 - 理论价格（中间价）
    # 如果我们以中间价交易是最理想的，但实际要以ask价买入
    df_execution['slippage'] = df_execution['execution_price'] - df_execution['mid_price']
    
    # 滑点成本 = 执行量 × 单位滑点
    # 这是因为以ask价而不是mid价买入而多付出的成本
    df_execution['slippage_cost'] = df_execution['execution_amount'] * df_execution['slippage']
    
    # 计算累积执行量（在每个区间内）
    # 先按interval_id和timestamp排序
    df_execution = df_execution.sort_values(['interval_id', 'timestamp'])
    
    # .groupby().cumsum()：分组累加
    # 在每个区间内，累加执行量，看已经完成了多少
    # 例如：第1笔5股，第2笔5股，第3笔5股 -> cumsum: 5, 10, 15
    df_execution['cumulative_execution_in_interval'] = df_execution.groupby('interval_id')['execution_amount'].cumsum()
    
    # 计算完成百分比
    # (已完成量 / 目标量) × 100 = 完成百分比
    # 例如：已完成500股，目标1000股 -> 50%
    df_execution['completion_pct'] = (df_execution['cumulative_execution_in_interval'] / 
                                       df_execution['target_execution_amount'] * 100)
    
    print(f"\n详细执行计划样例:")
    # 显示前15行的关键列
    print(df_execution[['timestamp', 'interval_label', 'execution_amount', 
                         'execution_price', 'execution_cost', 'slippage_cost',
                         'completion_pct']].head(15))
    
    return df_execution


def save_twap_results(interval_stats, df_execution):
    """
    保存TWAP执行结果到文件
    
    参数:
        interval_stats: 区间统计DataFrame
        df_execution: 详细执行计划DataFrame
    """
    print("\n保存执行计划...")
    
    # === 保存区间统计 ===
    
    # 1. 保存为parquet格式（高效的列式存储格式）
    # engine='pyarrow'：使用pyarrow引擎
    # compression='snappy'：使用snappy压缩算法（快速且压缩率好）
    # index=False：不保存索引列
    interval_stats.to_parquet('twap_interval_stats.parquet', 
                               engine='pyarrow', compression='snappy', index=False)
    
    # 2. 同时保存为CSV格式（便于用Excel查看）
    interval_stats.to_csv('twap_interval_stats.csv', index=False)
    
    # === 保存详细执行计划 ===
    
    # 保存完整的执行计划为parquet（包含所有列）
    df_execution.to_parquet('twap_execution_schedule.parquet', 
                             engine='pyarrow', compression='snappy', index=False)
    
    # 保存简化版CSV（只包含关键列，便于查看）
    # 用[列名列表]选择特定列
    df_execution[['timestamp', 'interval_label', 'execution_amount', 
                   'execution_price', 'execution_cost', 'slippage_cost',
                   'completion_pct']].to_csv('twap_execution_schedule.csv', index=False)
    
    # 打印保存的文件列表
    print("✓ 执行计划已保存:")
    print("  - twap_interval_stats.parquet")
    print("  - twap_interval_stats.csv")
    print("  - twap_execution_schedule.parquet")
    print("  - twap_execution_schedule.csv")


def visualize_execution_summary(df_execution, interval_stats):
    """
    打印执行摘要（汇总统计信息）
    
    参数:
        df_execution: 详细执行计划DataFrame
        interval_stats: 区间统计DataFrame
    """
    # 打印分隔线，让输出更美观
    print("\n" + "="*70)
    print("TWAP执行策略摘要")
    print("="*70)
    
    # === 【时间范围】 ===
    print(f"\n【时间范围】")
    # .min()取最小值（最早时间），.max()取最大值（最晚时间）
    print(f"  开始时间: {df_execution['timestamp'].min()}")
    print(f"  结束时间: {df_execution['timestamp'].max()}")
    # 两个时间相减得到时间差
    print(f"  总时长: {(df_execution['timestamp'].max() - df_execution['timestamp'].min())}")
    
    # === 【执行统计】 ===
    print(f"\n【执行统计】")
    # .nunique()计算不重复值的数量
    print(f"  总区间数: {interval_stats['interval_id'].nunique()}")
    print(f"  每区间时长: 10分钟")
    # len()返回行数，即总共有多少笔执行
    print(f"  总执行笔数: {len(df_execution)}")
    print(f"  执行频率: 每3秒")
    
    # === 【交易量统计】 ===
    print(f"\n【交易量统计】")
    # .iloc[-1]取最后一行（iloc是基于位置的索引）
    # amount是累积成交量，所以最后一个值就是全天的总成交量
    print(f"  总市场成交量: {df_execution['amount'].iloc[-1]:,.0f}")
    
    # .sum()对整列求和，得到我们的总执行量
    print(f"  总执行量: {df_execution['execution_amount'].sum():,.0f}")
    
    # 计算我们的执行量占市场总成交量的比例
    print(f"  执行占比: {df_execution['execution_amount'].sum() / df_execution['amount'].iloc[-1] * 100:.4f}%")
    
    # === 【成本分析】 ===
    print(f"\n【成本分析】")
    # 总执行成本 = 所有执行的成本之和
    print(f"  总执行成本: ${df_execution['execution_cost'].sum():,.2f}")
    
    # 总滑点成本 = 因为以ask价而不是mid价交易而多付的成本
    print(f"  总滑点成本: ${df_execution['slippage_cost'].sum():,.2f}")
    
    # .mean()计算平均值
    print(f"  平均执行价格: ${df_execution['execution_price'].mean():.2f}")
    
    # 计算平均滑点（以基点bp为单位，1bp = 0.01%）
    # (滑点 / 中间价) * 10000 = 基点数
    print(f"  平均滑点(bp): {(df_execution['slippage'] / df_execution['mid_price'] * 10000).mean():.2f}")
    
    print("\n" + "="*70)


def main():
    """
    主流程函数
    按顺序执行所有步骤，完成TWAP策略的计算
    """
    
    # === 步骤1: 读取采样数据 ===
    # 调用load_sampled_data函数，读取之前处理好的数据
    df = load_sampled_data('orderbook_sampled.parquet')
    
    # === 步骤2: 创建10分钟时间区间 ===
    # 给每条数据打上所属区间的标签
    # interval_minutes=10 表示每10分钟一个区间
    df = create_time_intervals(df, interval_minutes=10)
    
    # === 步骤3: 计算每个区间的统计指标 ===
    # 对每个10分钟区间，计算价格、成交量等统计信息
    interval_stats = calculate_interval_metrics(df)
    
    # === 步骤4: 计算TWAP执行策略 ===
    # execution_ratio=0.01 表示执行区间成交量的1%
    interval_stats = calculate_twap_execution(interval_stats, execution_ratio=0.01)
    
    # === 步骤5: 生成详细的执行计划 ===
    # 把区间级别的计划细化到每个时间点（每3秒）
    df_execution = generate_detailed_execution_schedule(df, interval_stats)
    
    # === 步骤6: 保存结果 ===
    # 把计算结果保存到文件
    save_twap_results(interval_stats, df_execution)
    
    # === 步骤7: 显示执行摘要 ===
    # 在控制台打印汇总统计信息
    visualize_execution_summary(df_execution, interval_stats)
    
    print("\n✓ TWAP策略计算完成！")
    
    # 返回两个结果DataFrame，供后续使用或分析
    return interval_stats, df_execution


# Python的标准写法：
# 如果这个文件被直接运行（而不是被import导入），就执行main()函数
# __name__ == '__main__' 只有在直接运行此文件时才为True
if __name__ == '__main__':
    # 调用main()函数，并接收返回的两个DataFrame
    interval_stats, df_execution = main()


