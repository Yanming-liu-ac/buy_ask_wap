# 修复 Matplotlib 导入错误

## 错误信息
```
ImportError: cannot import name 'LoadFlags' from 'matplotlib.ft2font'
```

## 原因
matplotlib库安装损坏或版本不兼容

## 解决方法

### 方法1：强制重新安装matplotlib（推荐）

在**PowerShell或CMD终端**（不是Jupyter notebook）中运行：

```bash
pip uninstall matplotlib -y
pip uninstall matplotlib-inline -y
pip install matplotlib
```

或者一次性执行：

```bash
pip install --upgrade --force-reinstall matplotlib
```

### 方法2：如果方法1不行，完全清理后重装

```bash
# 1. 卸载所有相关包
pip uninstall matplotlib matplotlib-inline kiwisolver pillow -y

# 2. 清理pip缓存
pip cache purge

# 3. 重新安装
pip install matplotlib
```

### 方法3：如果还是不行，升级pip后重试

```bash
# 升级pip
python -m pip install --upgrade pip

# 重新安装matplotlib
pip install --force-reinstall --no-cache-dir matplotlib
```

## 安装完成后

1. **关闭并重启Jupyter notebook**（重要！）
2. **重启kernel**
3. 重新运行导入matplotlib的cell

## 验证是否修复

在notebook中运行：

```python
import matplotlib
print(f"Matplotlib版本: {matplotlib.__version__}")

import matplotlib.pyplot as plt
print("✓ Matplotlib导入成功！")
```

如果能成功输出版本号，说明修复成功。

