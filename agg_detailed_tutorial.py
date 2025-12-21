"""
agg() 函数详解
用具体例子一步步理解 agg 的工作原理
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("agg() 函数详解 - 从简单到复杂")
print("=" * 70)

# ============================================================================
# 第一部分：准备示例数据
# ============================================================================
print("\n【第一部分：准备示例数据】\n")

# 创建一个简单的销售数据表
sales_data = pd.DataFrame({
    'product': ['苹果', '苹果', '苹果', '香蕉', '香蕉', '香蕉', '橙子', '橙子', '橙子'],
    'region': ['北京', '北京', '上海', '北京', '上海', '上海', '北京', '北京', '上海'],
    'price': [5.0, 5.5, 6.0, 3.0, 3.5, 3.2, 4.0, 4.5, 4.2],
    'quantity': [100, 150, 200, 80, 120, 100, 90, 110, 130]
})

print("原始数据：")
print(sales_data)
print()

# ============================================================================
# 第二部分：不用 agg，先理解分组
# ============================================================================
print("\n【第二部分：理解分组 groupby()】\n")

# groupby() 的作用：把数据按某列的值分成几组
grouped = sales_data.groupby('product')

print("按产品分组后，有几组？")
print(f"组数：{grouped.ngroups}组")
print()

print("每组包含哪些数据？")
for name, group in grouped:
    print(f"\n组名：{name}")
    print(group)

# ============================================================================
# 第三部分：单个统计函数
# ============================================================================
print("\n" + "=" * 70)
print("【第三部分：单个统计函数】")
print("=" * 70)

# 对分组后的数据，计算每组的平均值
print("\n1. 计算每个产品的平均价格：")
result1 = sales_data.groupby('product')['price'].mean()
print(result1)
print("\n解释：")
print("  苹果的平均价格 = (5.0 + 5.5 + 6.0) / 3 = 5.5")
print("  香蕉的平均价格 = (3.0 + 3.5 + 3.2) / 3 ≈ 3.23")
print("  橙子的平均价格 = (4.0 + 4.5 + 4.2) / 3 ≈ 4.23")

print("\n2. 计算每个产品的总销量：")
result2 = sales_data.groupby('product')['quantity'].sum()
print(result2)
print("\n解释：")
print("  苹果的总销量 = 100 + 150 + 200 = 450")
print("  香蕉的总销量 = 80 + 120 + 100 = 300")
print("  橙子的总销量 = 90 + 110 + 130 = 330")

# ============================================================================
# 第四部分：使用 agg() - 单列单个函数
# ============================================================================
print("\n" + "=" * 70)
print("【第四部分：使用 agg() - 单列单个函数】")
print("=" * 70)

print("\n上面的操作也可以用 agg() 来做：")
print("\n方式1：直接调用函数")
print("sales_data.groupby('product')['price'].mean()")
print(sales_data.groupby('product')['price'].mean())

print("\n方式2：使用 agg()")
print("sales_data.groupby('product')['price'].agg('mean')")
print(sales_data.groupby('product')['price'].agg('mean'))

print("\n两种方式结果完全一样！")

# ============================================================================
# 第五部分：使用 agg() - 单列多个函数（agg的优势开始体现）
# ============================================================================
print("\n" + "=" * 70)
print("【第五部分：使用 agg() - 单列多个函数】")
print("=" * 70)

print("\n如果我想同时计算平均值、最小值、最大值呢？")
print("\n不用 agg，需要写多次：")
print("  mean_val = df.groupby('product')['price'].mean()")
print("  min_val = df.groupby('product')['price'].min()")
print("  max_val = df.groupby('product')['price'].max()")

print("\n用 agg，一次搞定！")
result3 = sales_data.groupby('product')['price'].agg(['mean', 'min', 'max'])
print(result3)

print("\n解释：")
print("  苹果：平均5.5，最小5.0，最大6.0")
print("  香蕉：平均3.23，最小3.0，最大3.5")
print("  橙子：平均4.23，最小4.0，最大4.5")

# ============================================================================
# 第六部分：使用 agg() - 多列不同函数（这是最强大的功能！）
# ============================================================================
print("\n" + "=" * 70)
print("【第六部分：使用 agg() - 多列不同函数（最强大！）】")
print("=" * 70)

print("\n需求：对不同的列，应用不同的统计函数")
print("  - price列：计算平均值和标准差")
print("  - quantity列：计算总和和计数")
print()

# 这就是 agg() 的字典语法！
result4 = sales_data.groupby('product').agg({
    'price': ['mean', 'std'],      # price列计算均值和标准差
    'quantity': ['sum', 'count']   # quantity列计算总和和计数
})

print("结果：")
print(result4)
print()

print("详细解释：")
print("对于'苹果'这组：")
print("  price_mean = (5.0 + 5.5 + 6.0) / 3 = 5.5")
print("  price_std = 价格的标准差（衡量价格波动）≈ 0.5")
print("  quantity_sum = 100 + 150 + 200 = 450")
print("  quantity_count = 3（有3条记录）")

# ============================================================================
# 第七部分：理解结果的列名
# ============================================================================
print("\n" + "=" * 70)
print("【第七部分：理解结果的列名】")
print("=" * 70)

print("\nagg() 的结果会产生多层列名（MultiIndex）：")
print(f"列名：{result4.columns.tolist()}")
print()

print("访问特定列：")
print("\n方式1：用元组访问")
print("result4[('price', 'mean')]")
print(result4[('price', 'mean')])

print("\n方式2：扁平化列名（在代码中常用）")
result4_flat = sales_data.groupby('product').agg({
    'price': ['mean', 'std'],
    'quantity': ['sum', 'count']
})
# 扁平化列名：把 ('price', 'mean') 变成 'price_mean'
result4_flat.columns = ['_'.join(col) for col in result4_flat.columns]
print(result4_flat)

print("\n现在可以直接用简单的列名访问：")
print("result4_flat['price_mean']")
print(result4_flat['price_mean'])

# ============================================================================
# 第八部分：twap_execution.py 中的实际例子
# ============================================================================
print("\n" + "=" * 70)
print("【第八部分：twap_execution.py 中的实际例子】")
print("=" * 70)

print("\n创建模拟的订单薄数据：")
# 模拟10分钟区间的数据
orderbook_data = pd.DataFrame({
    'interval_id': [0, 0, 0, 1, 1, 1],
    'interval_label': ['10:00-10:10', '10:00-10:10', '10:00-10:10', 
                       '10:10-10:20', '10:10-10:20', '10:10-10:20'],
    'mid_price': [100.0, 100.5, 101.0, 101.5, 102.0, 101.8],
    'spread': [0.02, 0.03, 0.025, 0.028, 0.032, 0.030],
    'amount': [1000, 2000, 3000, 3500, 4200, 5000]
})

print(orderbook_data)

print("\n现在用 agg() 计算每个区间的统计指标：")
interval_stats = orderbook_data.groupby(['interval_id', 'interval_label']).agg({
    'mid_price': ['mean', 'std', 'min', 'max'],  # 价格的均值、标准差、最小、最大
    'spread': ['mean', 'std'],                    # 价差的均值、标准差
    'amount': ['first', 'last']                   # 成交量的第一个和最后一个值
})

print(interval_stats)

print("\n解释这个结果：")
print("\n区间 10:00-10:10 (interval_id=0):")
print("  mid_price_mean = (100.0 + 100.5 + 101.0) / 3 = 100.5")
print("  mid_price_min = 100.0")
print("  mid_price_max = 101.0")
print("  spread_mean = (0.02 + 0.03 + 0.025) / 3 = 0.025")
print("  amount_first = 1000 (区间开始时的累积成交量)")
print("  amount_last = 3000 (区间结束时的累积成交量)")

print("\n区间 10:10-10:20 (interval_id=1):")
print("  mid_price_mean = (101.5 + 102.0 + 101.8) / 3 ≈ 101.77")
print("  mid_price_min = 101.5")
print("  mid_price_max = 102.0")
print("  amount_first = 3500")
print("  amount_last = 5000")

# ============================================================================
# 第九部分：常用的聚合函数
# ============================================================================
print("\n" + "=" * 70)
print("【第九部分：常用的聚合函数列表】")
print("=" * 70)

print("""
可以在 agg() 中使用的常用函数：

统计类：
  'mean'   - 平均值
  'median' - 中位数
  'sum'    - 求和
  'count'  - 计数（非空值数量）
  'size'   - 大小（包括空值）
  
范围类：
  'min'    - 最小值
  'max'    - 最大值
  'std'    - 标准差（衡量波动性）
  'var'    - 方差
  
位置类：
  'first'  - 第一个值
  'last'   - 最后一个值
  'nth'    - 第n个值
  
其他：
  'nunique' - 不重复值的数量
  'prod'    - 乘积
  'quantile' - 分位数
""")

# ============================================================================
# 第十部分：自定义函数
# ============================================================================
print("\n" + "=" * 70)
print("【第十部分：使用自定义函数】")
print("=" * 70)

print("\nagg() 还可以使用自定义函数！")

# 定义一个自定义函数：计算范围（最大值-最小值）
def price_range(series):
    return series.max() - series.min()

result5 = sales_data.groupby('product')['price'].agg(['mean', price_range])
print(result5)

print("\n解释：")
print("  苹果的价格范围 = 6.0 - 5.0 = 1.0")
print("  香蕉的价格范围 = 3.5 - 3.0 = 0.5")
print("  橙子的价格范围 = 4.5 - 4.0 = 0.5")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("【总结：agg() 的三种主要用法】")
print("=" * 70)

print("""
1. 单列单个函数：
   df.groupby('category')['price'].agg('mean')
   
2. 单列多个函数：
   df.groupby('category')['price'].agg(['mean', 'std', 'min', 'max'])
   
3. 多列不同函数（最强大！）：
   df.groupby('category').agg({
       'price': ['mean', 'std'],
       'quantity': ['sum', 'count']
   })

核心思想：
  分组 (groupby) → 对每组应用函数 (agg) → 合并结果
  
  就像把学生按班级分组，然后计算每个班级的：
  - 数学成绩的平均分、最高分、最低分
  - 语文成绩的平均分、标准差
  - 学生人数
""")

print("\n" + "=" * 70)
print("运行完毕！建议：")
print("1. 仔细看每一部分的输出")
print("2. 理解数据如何分组")
print("3. 理解每个统计函数的含义")
print("4. 尝试修改代码，添加自己的数据和统计函数")
print("=" * 70)


