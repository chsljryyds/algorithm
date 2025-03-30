# 118----------leetcode   --------杨辉三角

给定一个非负整数 *`numRows`，*生成「杨辉三角」的前 *`numRows`* 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。





```python
def generate(numRows):
    if numRows == 0:
        return []
    
    # 初始化结果列表
    triangle = []
    
    # 第一行总是 [1]
    triangle.append([1])
    
    # 从第二行开始生成直到第 numRows 行
    for i in range(1, numRows):
        # 初始化当前行
        current_row = [0] * (i + 1)
        # 第一个元素总是 1
        current_row[0] = 1
        # 最后一个元素总是 1
        current_row[-1] = 1
        # 计算中间的元素
        for j in range(1, i):
            current_row[j] = triangle[i-1][j-1] + triangle[i-1][j]
        # 将当前行加入结果三角形
        triangle.append(current_row)
    
    return triangle

```

当生成杨辉三角的前 numRows 行时，每一行的数值是通过上一行计算得来的。这个问题可以通过迭代的方式来解决，每一行的计算都依赖于上一行的结果。

让我们逐步来讲解如何用 Python 来实现这个过程。

### 步骤分解

1. **初始化结果列表**：
   - 首先，我们创建一个空列表 `triangle`，用于存储生成的杨辉三角的结果。
2. **处理特殊情况**：
   - 如果 numRows 为 0，直接返回空列表。
3. **生成每一行**：
   - 从第一行开始生成，每一行的第一个和最后一个元素都是 1。
   - 对于中间的元素，根据杨辉三角的定义，它等于上一行中当前位置和前一个位置的和。
4. **迭代生成直到第 numRows 行**：
   - 使用一个循环从第二行开始到第 numRows 行。
   - 对于每一行，根据上一行的内容计算当前行的内容，并添加到 `triangle` 中。



### 解释代码

- **第一行初始化**：
  - 首先，我们创建一个空列表 `triangle`，并在其中添加第一行 `[1]`。
- **迭代生成每一行**：
  - 使用一个循环 `for i in range(1, numRows)` 来生成从第二行到第 numRows 行的内容。
  - 对于每一行，首先创建一个长度为 `i+1` 的列表 `current_row`，用来存储当前行的元素。
  - 设置 `current_row` 的第一个和最后一个元素为 1。
  - 对于中间的元素，通过遍历上一行 `triangle[i-1]` 的元素来计算，例如 `current_row[j] = triangle[i-1][j-1] + triangle[i-1][j]`。
  - 完成当前行的计算后，将其添加到 `triangle` 列表中。
- **返回结果**：
  - 最后返回生成的 `triangle` 列表，其中包含了杨辉三角的前 numRows 行。

### 示例

例如，调用 `generate(5)` 将会生成如下的杨辉三角：

```
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```

这个函数实现了一个高效的方法来生成杨辉三角，并且可以处理 numRows 为 0 的情况。