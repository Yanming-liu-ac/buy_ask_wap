"""
TWAP交易模拟器
实现带流动性约束的TWAP交易策略
"""

import pandas as pd
import numpy as np
from datetime import datetime


class TWAPSimulator:
    """
    TWAP交易模拟器
    
    参数:
        direction: 'buy' 或 'sell'
        emergency_cost_bps: 最后时刻强制成交的额外成本（基点，bp）
    """
    
    def __init__(self, direction='buy', emergency_cost_bps=5.0):
        self.direction = direction
        self.emergency_cost_bps = emergency_cost_bps
        self.results = []
    
    def simulate_interval(self, interval_data, target_volume):
        """
        模拟一个10分钟区间的TWAP交易
        
        参数:
            interval_data: 该区间的所有snapshot数据（DataFrame）
            target_volume: 目标交易量（股）
        
        返回:
            交易结果字典
        """
        interval_data = interval_data.sort_values('timestamp').reset_index(drop=True)
        n_snapshots = len(interval_data)
        
        if n_snapshots == 0:
            return None
        
        # 计算每个时刻的计划下单量（均匀分配）
        planned_volume_per_slice = target_volume / n_snapshots
        
        # 交易记录
        trades = []
        remaining_volume = target_volume
        cumulative_executed = 0
        
        for idx, row in interval_data.iterrows():
            is_last = (idx == n_snapshots - 1)
            
            # 如果是最后一个时刻，必须完成所有剩余订单
            if is_last:
                planned_volume = remaining_volume
            else:
                planned_volume = min(planned_volume_per_slice, remaining_volume)
            
            # 获取流动性限制
            if self.direction == 'buy':
                available_volume = row['ask1_v']  # 买入受卖一量限制
                price = row['ask1_p']  # 按卖一价成交
            else:  # sell
                available_volume = row['bid1_v']  # 卖出受买一量限制
                price = row['bid1_p']  # 按买一价成交
            
            # 实际可执行的量
            if is_last:
                # 最后时刻强制成交，即使超过流动性
                executed_volume = planned_volume
                is_emergency = (planned_volume > available_volume)
                
                if is_emergency:
                    # 计算额外成本
                    emergency_volume = planned_volume - available_volume
                    emergency_cost_multiplier = 1 + (self.emergency_cost_bps / 10000)
                    
                    if self.direction == 'buy':
                        emergency_price = price * emergency_cost_multiplier
                    else:
                        emergency_price = price / emergency_cost_multiplier
                    
                    # 加权平均价格
                    if planned_volume > 0:
                        weighted_price = (
                            available_volume * price + 
                            emergency_volume * emergency_price
                        ) / planned_volume
                    else:
                        weighted_price = price
                else:
                    weighted_price = price
                    emergency_volume = 0
            else:
                # 非最后时刻，受流动性限制
                executed_volume = min(planned_volume, available_volume)
                weighted_price = price
                is_emergency = False
                emergency_volume = 0
            
            # 记录交易
            if executed_volume > 0:
                trade = {
                    'timestamp': row['timestamp'],
                    'snapshot_idx': idx,
                    'planned_volume': planned_volume,
                    'executed_volume': executed_volume,
                    'price': weighted_price,
                    'mid_price': row['mid_price'],
                    'available_volume': available_volume,
                    'is_last': is_last,
                    'is_emergency': is_emergency,
                    'emergency_volume': emergency_volume
                }
                trades.append(trade)
                
                cumulative_executed += executed_volume
                remaining_volume -= executed_volume
        
        # 汇总结果
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            # 计算成交均价
            total_value = (trades_df['executed_volume'] * trades_df['price']).sum()
            avg_price = total_value / cumulative_executed if cumulative_executed > 0 else 0
            
            # TWAP基准价（mid_price的时间加权平均）
            twap_benchmark = interval_data['mid_price'].mean()
            
            # VWAP基准价（mid_price的成交量加权平均）
            # 使用区间内的volume变化作为权重
            interval_data_copy = interval_data.copy()
            interval_data_copy['volume_delta'] = interval_data_copy['volume'].diff().fillna(interval_data_copy['volume'])
            interval_data_copy['volume_delta'] = interval_data_copy['volume_delta'].abs()
            
            if interval_data_copy['volume_delta'].sum() > 0:
                vwap_benchmark = (
                    interval_data_copy['mid_price'] * interval_data_copy['volume_delta']
                ).sum() / interval_data_copy['volume_delta'].sum()
            else:
                vwap_benchmark = twap_benchmark
            
            # 滑点（基点）- 相对于TWAP
            if self.direction == 'buy':
                slippage_vs_twap_bps = ((avg_price - twap_benchmark) / twap_benchmark) * 10000
                slippage_vs_vwap_bps = ((avg_price - vwap_benchmark) / vwap_benchmark) * 10000
            else:
                slippage_vs_twap_bps = ((twap_benchmark - avg_price) / twap_benchmark) * 10000
                slippage_vs_vwap_bps = ((vwap_benchmark - avg_price) / vwap_benchmark) * 10000
            
            # 紧急成交统计
            emergency_trades = trades_df[trades_df['is_emergency']]
            total_emergency_volume = trades_df['emergency_volume'].sum()
            
            result = {
                'target_volume': target_volume,
                'executed_volume': cumulative_executed,
                'completion_rate': cumulative_executed / target_volume if target_volume > 0 else 0,
                'avg_execution_price': avg_price,
                'twap_benchmark': twap_benchmark,
                'vwap_benchmark': vwap_benchmark,
                'slippage_vs_twap_bps': slippage_vs_twap_bps,
                'slippage_vs_vwap_bps': slippage_vs_vwap_bps,
                'n_trades': len(trades_df),
                'n_emergency_trades': len(emergency_trades),
                'emergency_volume': total_emergency_volume,
                'emergency_volume_pct': (total_emergency_volume / target_volume * 100) if target_volume > 0 else 0,
                'trades': trades_df
            }
        else:
            result = {
                'target_volume': target_volume,
                'executed_volume': 0,
                'completion_rate': 0,
                'avg_execution_price': 0,
                'twap_benchmark': 0,
                'vwap_benchmark': 0,
                'slippage_vs_twap_bps': 0,
                'slippage_vs_vwap_bps': 0,
                'n_trades': 0,
                'n_emergency_trades': 0,
                'emergency_volume': 0,
                'emergency_volume_pct': 0,
                'trades': trades_df
            }
        
        return result
    
    def simulate_all_intervals(self, df, interval_stats):
        """
        模拟所有区间的TWAP交易
        """
        results = []
        
        for _, interval_info in interval_stats.iterrows():
            interval_id = interval_info['interval_id']
            target_volume = interval_info['target_order_volume']
            
            # 获取该区间的数据
            interval_data = df[df['interval_id'] == interval_id].copy()
            
            if len(interval_data) == 0:
                continue
            
            # 模拟交易
            result = self.simulate_interval(interval_data, target_volume)
            
            if result:
                result['interval_id'] = interval_id
                result['session'] = interval_info['session']
                results.append(result)
        
        self.results = results
        return results


def main():
    """主函数：运行TWAP模拟"""
    print("="*80)
    print("TWAP交易模拟")
    print("="*80)
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    df = pd.read_parquet('orderbook_sampled_10min_intervals.parquet')
    interval_stats = pd.read_parquet('interval_statistics.parquet')
    print(f"   ✓ 订单薄数据: {len(df)} 条")
    print(f"   ✓ 区间统计: {len(interval_stats)} 个区间")
    
    # 2. 创建模拟器并运行
    print("\n2. 运行TWAP模拟...")
    simulator = TWAPSimulator(direction='buy', emergency_cost_bps=5.0)
    results = simulator.simulate_all_intervals(df, interval_stats)
    print(f"   ✓ 模拟完成: {len(results)} 个区间")
    
    # 3. 整理结果
    print("\n3. 整理结果...")
    results_summary = pd.DataFrame([{
        'interval_id': r['interval_id'],
        'session': r['session'],
        'target_volume': r['target_volume'],
        'executed_volume': r['executed_volume'],
        'completion_rate': r['completion_rate'],
        'avg_execution_price': r['avg_execution_price'],
        'twap_benchmark': r['twap_benchmark'],
        'vwap_benchmark': r['vwap_benchmark'],
        'slippage_vs_twap_bps': r['slippage_vs_twap_bps'],
        'slippage_vs_vwap_bps': r['slippage_vs_vwap_bps'],
        'n_trades': r['n_trades'],
        'n_emergency_trades': r['n_emergency_trades'],
        'emergency_volume': r['emergency_volume'],
        'emergency_volume_pct': r['emergency_volume_pct']
    } for r in results])
    
    # 4. 统计分析
    print("\n4. 统计分析:")
    print(f"   - 平均完成率: {results_summary['completion_rate'].mean()*100:.2f}%")
    print(f"   - 平均滑点 vs TWAP: {results_summary['slippage_vs_twap_bps'].mean():.2f} bp")
    print(f"   - 平均滑点 vs VWAP: {results_summary['slippage_vs_vwap_bps'].mean():.2f} bp")
    print(f"   - 紧急成交比例: {results_summary['emergency_volume'].sum() / results_summary['target_volume'].sum() * 100:.2f}%")
    print(f"   - 有紧急成交的区间: {(results_summary['n_emergency_trades'] > 0).sum()} 个")
    
    # TWAP vs VWAP 比较
    print("\n5. TWAP vs VWAP 基准对比:")
    print(f"   - TWAP基准均价: {results_summary['twap_benchmark'].mean():.4f}")
    print(f"   - VWAP基准均价: {results_summary['vwap_benchmark'].mean():.4f}")
    price_diff_bps = ((results_summary['vwap_benchmark'].mean() - results_summary['twap_benchmark'].mean()) / 
                       results_summary['twap_benchmark'].mean()) * 10000
    print(f"   - VWAP与TWAP差异: {price_diff_bps:.2f} bp")
    
    # 哪个基准更优
    better_vs_twap = (results_summary['slippage_vs_twap_bps'].abs().mean())
    better_vs_vwap = (results_summary['slippage_vs_vwap_bps'].abs().mean())
    print(f"\n6. 基准选择建议:")
    if better_vs_vwap < better_vs_twap:
        print(f"   ✓ VWAP更接近实际成交价（平均偏差{better_vs_vwap:.2f}bp vs {better_vs_twap:.2f}bp）")
        print(f"   → 建议使用VWAP作为业绩基准")
    else:
        print(f"   ✓ TWAP更接近实际成交价（平均偏差{better_vs_twap:.2f}bp vs {better_vs_vwap:.2f}bp）")
        print(f"   → 建议使用TWAP作为业绩基准")
    
    # 5. 保存结果
    print("\n7. 保存结果...")
    results_summary.to_csv('twap_simulation_summary.csv', index=False, encoding='utf-8-sig')
    results_summary.to_parquet('twap_simulation_summary.parquet', index=False)
    print(f"   ✓ 已保存: twap_simulation_summary.csv")
    print(f"   ✓ 已保存: twap_simulation_summary.parquet")
    
    print("\n" + "="*80)
    print("✓ TWAP模拟完成！")
    print("="*80)
    
    return results_summary, results


if __name__ == '__main__':
    results_summary, results = main()

