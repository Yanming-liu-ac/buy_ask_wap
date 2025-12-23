"""
测试预测质量 - 对比线性回归 vs 简单基准方法
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("预测质量测试")
print("=" * 80)

# 1. 加载数据
df = pd.read_parquet('orderbook_sampled_10min_intervals.parquet')
df = df.sort_values('timestamp').reset_index(drop=True)

# 预测目标：60秒后的价格
forecast_horizon = 20
df['target'] = df['mid_price'].shift(-forecast_horizon)
df = df.dropna()

# 划分数据
n = len(df)
train_size = int(n * 0.85)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f"\n数据划分:")
print(f"  训练集: {len(train)}")
print(f"  测试集: {len(test)}")

# ============================================================================
# 基准方法1: 简单持续 (Naive Persistence)
# ============================================================================
print("\n\n[基准1] 简单持续: 未来价格 = 当前价格")
test['pred_naive'] = test['mid_price']

mae_naive = mean_absolute_error(test['target'], test['pred_naive'])
rmse_naive = np.sqrt(mean_squared_error(test['target'], test['pred_naive']))

# 方向准确率
direction_naive = ((test['target'] - test['mid_price']) * 
                   (test['pred_naive'] - test['mid_price']) > 0).mean() * 100

print(f"  MAE:  {mae_naive:.4f}")
print(f"  RMSE: {rmse_naive:.4f}")
print(f"  方向准确率: {direction_naive:.2f}%")

# ============================================================================
# 基准方法2: 移动平均
# ============================================================================
print("\n[基准2] 移动平均: 未来价格 = 过去5个点的平均")
test['pred_ma'] = test['mid_price'].rolling(5).mean()
test = test.dropna()

mae_ma = mean_absolute_error(test['target'], test['pred_ma'])
rmse_ma = np.sqrt(mean_squared_error(test['target'], test['pred_ma']))

direction_ma = ((test['target'].diff() > 0) == 
                (test['pred_ma'].diff() > 0)).mean() * 100

print(f"  MAE:  {mae_ma:.4f}")
print(f"  RMSE: {rmse_ma:.4f}")
print(f"  方向准确率: {direction_ma:.2f}%")

# ============================================================================
# 基准方法3: 趋势延续
# ============================================================================
print("\n[基准3] 趋势延续: 未来变化 = 过去变化")
test['price_change_past'] = test['mid_price'].diff(5)  # 过去15秒变化
test['pred_trend'] = test['mid_price'] + test['price_change_past']
test = test.dropna()

mae_trend = mean_absolute_error(test['target'], test['pred_trend'])
rmse_trend = np.sqrt(mean_squared_error(test['target'], test['pred_trend']))

direction_trend = ((test['target'] - test['mid_price']) * 
                   (test['pred_trend'] - test['mid_price']) > 0).mean() * 100

print(f"  MAE:  {mae_trend:.4f}")
print(f"  RMSE: {rmse_trend:.4f}")
print(f"  方向准确率: {direction_trend:.2f}%")

# ============================================================================
# 对比：线性回归
# ============================================================================
print("\n[你的模型] 线性回归")
print(f"  MAE:  0.2198")
print(f"  RMSE: 0.2745")
print(f"  方向准确率: 52.36%")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("结论")
print("=" * 80)

print("\n如果线性回归的MAE和基准方法差不多，那说明：")
print("1. 模型没有学到真正的预测规律")
print("2. 只是在做'平滑'，不是'预测'")
print("3. 价格本身就很难预测（市场有效性）")

print("\n方向准确率 ~50% 意味着：")
print("- 模型无法判断价格会涨还是跌")
print("- 这是金融市场预测的常见问题")
print("- 需要更复杂的特征或模型")

# ============================================================================
# 可视化对比
# ============================================================================
print("\n\n生成可视化对比...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 预测结果对比
sample = test.iloc[:100]
axes[0, 0].plot(sample.index, sample['target'], label='真实值', linewidth=2)
axes[0, 0].plot(sample.index, sample['pred_naive'], label='简单持续', alpha=0.7)
axes[0, 0].plot(sample.index, sample['pred_ma'], label='移动平均', alpha=0.7)
axes[0, 0].set_title('预测方法对比（前100个点）', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 子图2: MAE对比
methods = ['简单持续', '移动平均', '趋势延续', '线性回归']
maes = [mae_naive, mae_ma, mae_trend, 0.2198]
axes[0, 1].bar(methods, maes, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('MAE对比 (越小越好)', fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='y')

# 子图3: 方向准确率对比
directions = [direction_naive, direction_ma, direction_trend, 52.36]
colors = ['red' if d < 55 else 'green' for d in directions]
axes[1, 0].bar(methods, directions, alpha=0.7, color=colors, edgecolor='black')
axes[1, 0].axhline(y=50, color='red', linestyle='--', label='随机猜测')
axes[1, 0].set_ylabel('方向准确率 (%)')
axes[1, 0].set_title('方向准确率对比 (>55%才算好)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# 子图4: 误差分布
axes[1, 1].hist(test['target'] - test['pred_naive'], bins=30, 
                alpha=0.5, label='简单持续', edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('预测误差')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title('误差分布', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_quality_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] 保存: prediction_quality_comparison.png")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

plt.show()

