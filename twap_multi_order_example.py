"""
使用类的好处和多订单执行示例

使用类(Class)的好处：
1. 封装性：将相关的数据和方法组织在一起
2. 可复用性：可以创建多个不同配置的模拟器实例
3. 状态管理：每个实例维护自己的状态（direction, results等）
4. 扩展性：容易添加新功能和方法
5. 对比实验：可以同时运行多个不同策略的模拟器
"""

from twap_simulator import TWAPSimulator
import pandas as pd
import numpy as np


# ============================================================================
# 示例1：使用类的好处 - 创建不同配置的模拟器
# ============================================================================

def example_1_multiple_configurations():
    """
    演示：创建多个不同配置的模拟器进行对比实验
    """
    print("="*80)
    print("示例1：不同配置对比实验")
    print("="*80)
    
    # 读取数据
    df = pd.read_parquet('orderbook_sampled_10min_intervals.parquet')
    interval_stats = pd.read_parquet('interval_statistics.parquet')
    
    # 创建多个不同配置的模拟器
    simulators = {
        '买入-低成本': TWAPSimulator(direction='buy', emergency_cost_bps=3.0),
        '买入-中成本': TWAPSimulator(direction='buy', emergency_cost_bps=5.0),
        '买入-高成本': TWAPSimulator(direction='buy', emergency_cost_bps=10.0),
        '卖出-中成本': TWAPSimulator(direction='sell', emergency_cost_bps=5.0),
    }
    
    # 运行所有模拟器
    all_results = {}
    for name, simulator in simulators.items():
        print(f"\n运行模拟器: {name}")
        results = simulator.simulate_all_intervals(df, interval_stats)
        all_results[name] = results
        
        # 汇总统计
        results_df = pd.DataFrame([{
            'interval_id': r['interval_id'],
            'slippage_vs_twap': r['slippage_vs_twap_bps'],
            'emergency_pct': r['emergency_volume_pct']
        } for r in results])
        
        print(f"  平均滑点: {results_df['slippage_vs_twap'].mean():.3f} bp")
        print(f"  紧急成交比例: {results_df['emergency_pct'].mean():.2f}%")
    
    return all_results


# ============================================================================
# 示例2：处理多个订单 - 按顺序执行
# ============================================================================

def example_2_multiple_orders_sequential():
    """
    演示：处理多个订单，按顺序执行
    """
    print("\n" + "="*80)
    print("示例2：多个订单顺序执行")
    print("="*80)
    
    # 读取数据
    df = pd.read_parquet('orderbook_sampled_10min_intervals.parquet')
    interval_stats = pd.read_parquet('interval_statistics.parquet')
    
    # 定义多个订单
    orders = [
        {'order_id': 'ORDER_001', 'direction': 'buy', 'volume': 1000, 'intervals': [1, 2, 3]},
        {'order_id': 'ORDER_002', 'direction': 'buy', 'volume': 2000, 'intervals': [4, 5]},
        {'order_id': 'ORDER_003', 'direction': 'sell', 'volume': 1500, 'intervals': [6, 7, 8]},
    ]
    
    all_order_results = []
    
    for order in orders:
        print(f"\n处理订单: {order['order_id']}")
        print(f"  方向: {order['direction']}")
        print(f"  总量: {order['volume']} 股")
        print(f"  执行区间: {order['intervals']}")
        
        # 为这个订单创建模拟器
        simulator = TWAPSimulator(
            direction=order['direction'], 
            emergency_cost_bps=5.0
        )
        
        # 分配到各区间的量
        volume_per_interval = order['volume'] / len(order['intervals'])
        
        order_results = []
        for interval_id in order['intervals']:
            # 获取该区间数据
            interval_data = df[df['interval_id'] == interval_id].copy()
            
            # 模拟该区间的交易
            result = simulator.simulate_interval(interval_data, volume_per_interval)
            
            if result:
                result['order_id'] = order['order_id']
                result['interval_id'] = interval_id
                order_results.append(result)
        
        # 汇总该订单的结果
        total_executed = sum(r['executed_volume'] for r in order_results)
        avg_price = sum(r['avg_execution_price'] * r['executed_volume'] 
                       for r in order_results) / total_executed if total_executed > 0 else 0
        
        print(f"  实际成交: {total_executed:.0f} 股")
        print(f"  成交均价: {avg_price:.4f}")
        print(f"  完成率: {total_executed / order['volume'] * 100:.2f}%")
        
        all_order_results.extend(order_results)
    
    return all_order_results


# ============================================================================
# 示例3：处理多个订单 - 同时执行（Portfolio级别）
# ============================================================================

class PortfolioTWAPSimulator:
    """
    Portfolio级别的TWAP模拟器
    可以同时处理多个股票/订单
    """
    
    def __init__(self):
        self.orders = []
        self.simulators = {}
    
    def add_order(self, order_id, symbol, direction, target_volume, 
                  interval_ids, emergency_cost_bps=5.0):
        """添加一个订单"""
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'target_volume': target_volume,
            'interval_ids': interval_ids,
            'emergency_cost_bps': emergency_cost_bps
        }
        self.orders.append(order)
        
        # 为每个订单创建独立的模拟器
        self.simulators[order_id] = TWAPSimulator(
            direction=direction,
            emergency_cost_bps=emergency_cost_bps
        )
    
    def execute_all_orders(self, df_dict):
        """
        执行所有订单
        
        参数:
            df_dict: {symbol: DataFrame} 字典，每个symbol的订单薄数据
        """
        all_results = {}
        
        for order in self.orders:
            order_id = order['order_id']
            symbol = order['symbol']
            
            print(f"\n执行订单: {order_id} ({symbol})")
            
            # 获取该股票的数据
            df = df_dict.get(symbol)
            if df is None:
                print(f"  警告: 找不到{symbol}的数据")
                continue
            
            simulator = self.simulators[order_id]
            order_results = []
            
            # 分配到各区间
            volume_per_interval = order['target_volume'] / len(order['interval_ids'])
            
            for interval_id in order['interval_ids']:
                interval_data = df[df['interval_id'] == interval_id].copy()
                result = simulator.simulate_interval(interval_data, volume_per_interval)
                
                if result:
                    result['order_id'] = order_id
                    result['symbol'] = symbol
                    order_results.append(result)
            
            all_results[order_id] = order_results
            
            # 输出该订单汇总
            if order_results:
                total_exec = sum(r['executed_volume'] for r in order_results)
                print(f"  总成交量: {total_exec:.0f} / {order['target_volume']:.0f} 股")
        
        return all_results
    
    def get_summary(self, all_results):
        """生成所有订单的汇总报告"""
        summary = []
        
        for order in self.orders:
            order_id = order['order_id']
            results = all_results.get(order_id, [])
            
            if results:
                total_target = order['target_volume']
                total_executed = sum(r['executed_volume'] for r in results)
                avg_price = (sum(r['avg_execution_price'] * r['executed_volume'] 
                            for r in results) / total_executed if total_executed > 0 else 0)
                avg_slippage = np.mean([r['slippage_vs_twap_bps'] for r in results])
                
                summary.append({
                    'order_id': order_id,
                    'symbol': order['symbol'],
                    'direction': order['direction'],
                    'target_volume': total_target,
                    'executed_volume': total_executed,
                    'completion_rate': total_executed / total_target,
                    'avg_price': avg_price,
                    'avg_slippage_bps': avg_slippage
                })
        
        return pd.DataFrame(summary)


def example_3_portfolio_execution():
    """
    演示：Portfolio级别的多订单执行
    """
    print("\n" + "="*80)
    print("示例3：Portfolio级别多订单执行")
    print("="*80)
    
    # 创建Portfolio模拟器
    portfolio_sim = PortfolioTWAPSimulator()
    
    # 添加多个订单（可以是不同股票）
    portfolio_sim.add_order(
        order_id='ORD001',
        symbol='TEST001',
        direction='buy',
        target_volume=5000,
        interval_ids=[1, 2, 3, 4, 5],
        emergency_cost_bps=5.0
    )
    
    portfolio_sim.add_order(
        order_id='ORD002',
        symbol='TEST001',  # 同一个股票的另一个订单
        direction='sell',
        target_volume=3000,
        interval_ids=[10, 11, 12],
        emergency_cost_bps=5.0
    )
    
    # 读取数据
    df = pd.read_parquet('orderbook_sampled_10min_intervals.parquet')
    df_dict = {'TEST001': df}  # 实际应用中可能有多个股票
    
    # 执行所有订单
    all_results = portfolio_sim.execute_all_orders(df_dict)
    
    # 生成汇总报告
    summary = portfolio_sim.get_summary(all_results)
    print("\n" + "="*80)
    print("Portfolio执行汇总:")
    print("="*80)
    print(summary)
    
    return portfolio_sim, all_results, summary


# ============================================================================
# 主函数：演示所有示例
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("🎯 TWAP模拟器 - 使用类的好处和多订单执行演示")
    print("="*80)
    
    # 示例1：不同配置对比
    print("\n📊 场景1：对比不同紧急成本配置的影响")
    results_1 = example_1_multiple_configurations()
    
    # 示例2：多个订单顺序执行
    print("\n📊 场景2：多个订单按顺序执行")
    results_2 = example_2_multiple_orders_sequential()
    
    # 示例3：Portfolio级别执行
    print("\n📊 场景3：Portfolio级别同时管理多个订单")
    portfolio_sim, results_3, summary_3 = example_3_portfolio_execution()
    
    print("\n" + "="*80)
    print("✅ 所有示例演示完成！")
    print("="*80)
    
    print("\n💡 使用类的主要好处总结：")
    print("  1. 状态隔离：每个模拟器实例有独立的配置和状态")
    print("  2. 代码复用：同样的代码可以处理多个订单")
    print("  3. 灵活配置：可以为不同订单设置不同的参数")
    print("  4. 易于扩展：可以继承类来添加新功能（如PortfolioTWAPSimulator）")
    print("  5. 对比实验：可以同时运行多种策略进行比较")


if __name__ == '__main__':
    main()

