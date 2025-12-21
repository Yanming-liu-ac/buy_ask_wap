# Pandas 核心概念与语法详解

这个文档详细解释在 `twap_execution.py` 中用到的pandas概念和语法。

## 目录
1. [DataFrame基础](#dataframe基础)
2. [列操作](#列操作)
3. [时间处理](#时间处理)
4. [分组聚合](#分组聚合)
5. [数据合并](#数据合并)
6. [常用函数](#常用函数)
7. [代码示例对照](#代码示例对照)

---

## DataFrame基础

### 什么是DataFrame？

DataFrame是pandas中最核心的数据结构，可以理解为：
- **类似Excel表格**：有行和列
- **类似数据库表**：每列有列名，每行有索引
- **类似字典**：可以用列名访问数据

```python
import pandas as pd

# 创建一个简单的DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

print(df)
#       name  age  salary
# 0    Alice   25   50000
# 1      Bob   30   60000
# 2  Charlie   35   70000
```

### 基本操作

```python
# 查看前几行
df.head()      # 默认前5行
df.head(10)    # 前10行

# 查看行数和列数
len(df)        # 行数
df.shape       # (行数, 列数)，例如 (3, 3)

# 查看列名
df.columns     # Index(['name', 'age', 'salary'])

# 查看数据类型
df.dtypes      # 每列的数据类型

# 查看统计信息
df.describe()  # 数值列的统计摘要
```

---

## 列操作

### 访问列

```python
# 方式1：用方括号（推荐）
df['age']               # 获取age列（返回Series）
df[['name', 'age']]     # 获取多列（返回DataFrame）

# 方式2：用点号（只适用于列名没有空格的情况）
df.age                  # 同 df['age']
```

### 创建新列

```python
# 直接赋值创建新列
df['bonus'] = df['salary'] * 0.1

# 基于条件创建
df['is_senior'] = df['age'] > 30

# 基于多列计算
df['total_comp'] = df['salary'] + df['bonus']
```

### 向量化操作（重要！）

pandas的强大之处在于**向量化操作**：对整列操作，而不是循环每个元素。

```python
# ❌ 慢的做法（Python循环）
result = []
for i in range(len(df)):
    result.append(df['salary'][i] * 1.1)
df['new_salary'] = result

# ✅ 快的做法（向量化）
df['new_salary'] = df['salary'] * 1.1

# 更多向量化操作
df['age'] + 5              # 每个值加5
df['salary'] * 1.1         # 每个值乘1.1
df['age'] > 30             # 返回布尔值Series
df['name'].str.upper()     # 字符串操作
```

**在 twap_execution.py 中的例子：**
```python
# 第132-133行：向量化计算目标执行量
interval_stats['target_execution_amount'] = (
    interval_stats['interval_amount'] * execution_ratio
)
# 这一行代码对整列的每个值都乘以 execution_ratio
```

---

## 时间处理

### datetime类型

```python
# 转换为datetime类型
df['date'] = pd.to_datetime(df['date_string'])

# 提取时间组件（使用 .dt 访问器）
df['year'] = df['date'].dt.year       # 年份
df['month'] = df['date'].dt.month     # 月份
df['day'] = df['date'].dt.day         # 日期
df['hour'] = df['date'].dt.hour       # 小时
df['minute'] = df['date'].dt.minute   # 分钟
df['weekday'] = df['date'].dt.weekday # 星期几（0=周一）
```

**在 twap_execution.py 中的例子：**
```python
# 第50-51行：提取时间组件
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

# 第24行：转换为datetime类型
df['timestamp'] = pd.to_datetime(df['timestamp'])
```

### 时间筛选

```python
# 筛选特定时间范围
df[df['timestamp'] >= '2025-12-20 10:00:00']
df[(df['timestamp'] >= '10:00') & (df['timestamp'] < '15:00')]
```

---

## 分组聚合

### groupby() - pandas最强大的功能之一

`groupby()` 的工作流程：**分组 → 应用函数 → 合并结果**

```python
# 基本语法
df.groupby('category').mean()        # 按category分组，计算每组的平均值
df.groupby('category')['price'].sum()  # 按category分组，只对price列求和

# 实际例子
sales_df = pd.DataFrame({
    'product': ['A', 'B', 'A', 'B', 'A'],
    'region': ['East', 'East', 'West', 'West', 'East'],
    'sales': [100, 200, 150, 250, 120]
})

# 按产品分组，计算总销售额
sales_df.groupby('product')['sales'].sum()
# product
# A    370
# B    450

# 按多列分组
sales_df.groupby(['product', 'region'])['sales'].sum()
# product  region
# A        East      220
#          West      150
# B        East      200
#          West      250
```

### agg() - 灵活的聚合

```python
# 单列多种聚合
df.groupby('category')['price'].agg(['mean', 'std', 'min', 'max'])

# 多列不同聚合
df.groupby('category').agg({
    'price': ['mean', 'std'],
    'quantity': 'sum',
    'date': ['min', 'max']
})
```

**在 twap_execution.py 中的例子：**
```python
# 第90-97行：复杂的分组聚合
interval_stats = df.groupby(['interval_id', 'interval_label']).agg({
    'timestamp': ['min', 'max', 'count'],
    'amount': ['first', 'last'],
    'mid_price': ['mean', 'std', 'min', 'max'],
    'spread': ['mean', 'std'],
    'spread_bps': 'mean',
    'volume': ['first', 'last'],
}).reset_index()
```

这段代码的含义：
1. 按 `interval_id` 和 `interval_label` 分组
2. 对每组的不同列应用不同的统计函数
3. `reset_index()` 把分组键变回普通列

### 分组内累积计算

```python
# 分组内累加
df.groupby('group')['value'].cumsum()

# 分组内累积均值
df.groupby('group')['value'].expanding().mean()
```

**在 twap_execution.py 中的例子：**
```python
# 第201行：在每个区间内累加执行量
df_execution['cumulative_execution_in_interval'] = (
    df_execution.groupby('interval_id')['execution_amount'].cumsum()
)
```

---

## 数据合并

### merge() - 类似SQL的JOIN

```python
# 基本用法
df1.merge(df2, on='key_column', how='left')

# how参数的选项：
# - 'left': 左连接（保留df1的所有行）
# - 'right': 右连接（保留df2的所有行）
# - 'inner': 内连接（只保留匹配的行）
# - 'outer': 外连接（保留所有行）

# 示例
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

salaries = pd.DataFrame({
    'emp_id': [1, 2, 4],
    'salary': [50000, 60000, 70000]
})

# 左连接
result = employees.merge(salaries, on='emp_id', how='left')
#    emp_id     name   salary
# 0       1    Alice  50000.0
# 1       2      Bob  60000.0
# 2       3  Charlie      NaN  # Charlie没有工资记录
```

**在 twap_execution.py 中的例子：**
```python
# 第178-183行：合并区间统计到每个时间点
df_execution = df.merge(
    interval_stats[['interval_id', 'target_execution_amount', 
                    'execution_per_snapshot', 'timestamp_count']], 
    on='interval_id', 
    how='left'
)
```

---

## 常用函数

### 统计函数

```python
df['column'].sum()      # 求和
df['column'].mean()     # 平均值
df['column'].median()   # 中位数
df['column'].std()      # 标准差
df['column'].var()      # 方差
df['column'].min()      # 最小值
df['column'].max()      # 最大值
df['column'].count()    # 计数（非空值）
df['column'].nunique()  # 不重复值的数量
```

### 排序

```python
# 按列排序
df.sort_values('age')                    # 升序
df.sort_values('age', ascending=False)   # 降序
df.sort_values(['age', 'salary'])        # 多列排序
```

**在 twap_execution.py 中的例子：**
```python
# 第47行：按时间排序
df = df.sort_values('timestamp').copy()

# 第200行：按多列排序
df_execution = df_execution.sort_values(['interval_id', 'timestamp'])
```

### 筛选

```python
# 条件筛选
df[df['age'] > 30]                      # 单条件
df[(df['age'] > 25) & (df['salary'] > 50000)]  # 多条件（与）
df[(df['age'] < 25) | (df['age'] > 60)]        # 多条件（或）

# 注意：& 是"与"，| 是"或"，每个条件要用括号括起来
```

### 文件读写

```python
# 读取
df = pd.read_csv('data.csv')
df = pd.read_parquet('data.parquet')
df = pd.read_excel('data.xlsx')

# 写入
df.to_csv('output.csv', index=False)
df.to_parquet('output.parquet', index=False)
df.to_excel('output.xlsx', index=False)
```

---

## 代码示例对照

### 示例1：时间分组

**目标**：把每天的数据按10分钟分组

```python
# 计算从0点开始的分钟数
df['minutes_from_start'] = df['hour'] * 60 + df['minute']

# 创建10分钟区间编号
df['interval_id'] = df['minutes_from_start'] // 10

# 解释：
# 10:05 → 605分钟 → 605 // 10 = 60（第60个区间）
# 10:08 → 608分钟 → 608 // 10 = 60（第60个区间）
# 10:12 → 612分钟 → 612 // 10 = 61（第61个区间）
```

### 示例2：计算价差

```python
# 订单薄数据
orderbook = pd.DataFrame({
    'bid1_p': [99.5, 100.0, 100.5],  # 买一价
    'ask1_p': [100.0, 100.5, 101.0]  # 卖一价
})

# 向量化计算价差（整列操作）
orderbook['spread'] = orderbook['ask1_p'] - orderbook['bid1_p']
# spread: [0.5, 0.5, 0.5]

# 向量化计算中间价
orderbook['mid_price'] = (orderbook['ask1_p'] + orderbook['bid1_p']) / 2
# mid_price: [99.75, 100.25, 100.75]
```

### 示例3：TWAP执行量分配

```python
# 假设我们有区间统计
interval_stats = pd.DataFrame({
    'interval_id': [0, 1, 2],
    'interval_amount': [100000, 150000, 120000],  # 区间成交量
    'timestamp_count': [200, 200, 200]  # 每个区间有200个数据点
})

# 计算目标执行量（1%）
interval_stats['target'] = interval_stats['interval_amount'] * 0.01
# target: [1000, 1500, 1200]

# 均匀分配到每个时间点
interval_stats['per_snapshot'] = (
    interval_stats['target'] / interval_stats['timestamp_count']
)
# per_snapshot: [5, 7.5, 6]
```

### 示例4：分组累加

```python
# 执行记录
execution = pd.DataFrame({
    'interval_id': [0, 0, 0, 1, 1, 1],
    'execution_amount': [5, 5, 5, 7.5, 7.5, 7.5]
})

# 在每个区间内累加
execution['cumulative'] = (
    execution.groupby('interval_id')['execution_amount'].cumsum()
)

print(execution)
#    interval_id  execution_amount  cumulative
# 0            0               5.0         5.0
# 1            0               5.0        10.0
# 2            0               5.0        15.0
# 3            1               7.5         7.5
# 4            1               7.5        15.0
# 5            1               7.5        22.5
```

---

## 关键概念总结

### 1. 向量化 vs 循环

```python
# ❌ 避免使用循环
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col1'] * 2

# ✅ 使用向量化
df['new_col'] = df['col1'] * 2
```

### 2. 链式操作

```python
# 可以连续调用多个方法
result = (df
    .sort_values('timestamp')
    .groupby('category')
    .agg({'price': 'mean'})
    .reset_index()
)
```

### 3. copy() 的重要性

```python
# 创建副本，避免修改原数据
df_new = df.copy()

# 如果不用copy()，可能会意外修改原数据
df_new = df  # 这只是创建了引用，不是副本！
```

### 4. 常用的方法链

```python
(df
    .dropna()              # 删除空值
    .sort_values('col')    # 排序
    .reset_index(drop=True)  # 重置索引
    .head(10)              # 取前10行
)
```

---

## 调试技巧

```python
# 1. 查看数据类型
print(df.dtypes)

# 2. 查看缺失值
print(df.isnull().sum())

# 3. 查看唯一值
print(df['category'].unique())
print(df['category'].nunique())

# 4. 查看统计摘要
print(df.describe())

# 5. 查看前几行
print(df.head())

# 6. 查看列名
print(df.columns)

# 7. 查看形状
print(df.shape)  # (行数, 列数)
```

---

## 学习建议

1. **先理解概念，再记语法**：理解DataFrame是什么，比记住所有函数更重要
2. **多用向量化操作**：这是pandas高效的关键
3. **善用 .head() 和 print()**：经常查看中间结果
4. **从简单例子开始**：先用小数据集测试，理解后再应用到大数据
5. **查阅官方文档**：pandas.pydata.org 有详细的文档和示例

---

## 扩展阅读

- [Pandas官方教程](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
- [10分钟入门Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas速查表](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

