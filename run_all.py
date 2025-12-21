"""
一键运行所有流程
"""
import subprocess
import sys


def run_script(script_name, description):
    """运行Python脚本"""
    print("\n" + "="*70)
    print(f"执行: {description}")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, script_name], 
                          capture_output=False, 
                          text=True)
    
    if result.returncode != 0:
        print(f"\n❌ 错误: {script_name} 执行失败")
        sys.exit(1)
    
    print(f"\n✓ {script_name} 执行成功")


def main():
    """主流程"""
    print("开始运行完整流程...")
    print("这个过程包括：数据生成 -> 数据处理 -> TWAP策略计算")
    
    # 步骤1: 生成订单薄数据
    run_script('generate_orderbook_data.py', '步骤1: 生成订单薄snapshot数据')
    
    # 步骤2: 处理数据并采样
    run_script('process_orderbook_data.py', '步骤2: 数据采样和价差计算')
    
    # 步骤3: 计算TWAP执行策略
    run_script('twap_execution.py', '步骤3: TWAP策略计算')
    
    print("\n" + "="*70)
    print("✓ 所有流程执行完成！")
    print("="*70)
    
    print("\n生成的文件:")
    print("  1. orderbook_snapshots_20251220.parquet - 原始订单薄数据")
    print("  2. orderbook_sampled.parquet - 采样后的数据")
    print("  3. twap_interval_stats.parquet/csv - 区间统计")
    print("  4. twap_execution_schedule.parquet/csv - 详细执行计划")
    
    print("\n您可以使用以下代码读取结果:")
    print("\n  import pandas as pd")
    print("  df = pd.read_parquet('twap_execution_schedule.parquet')")
    print("  print(df.head())")


if __name__ == '__main__':
    main()

