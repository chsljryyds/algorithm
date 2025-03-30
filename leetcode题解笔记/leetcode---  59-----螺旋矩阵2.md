# leetcode---  59-----螺旋矩阵2

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。



**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/13/spiraln.jpg)

```
输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]
```

**示例 2：**

```
输入：n = 1
输出：[[1]]
```



解法：

```python
def generateMatrix(n):
    # 初始化一个 n x n 的矩阵，并用零填充
    matrix = [[0] * n for _ in range(n)]
    num = 1  # 从1开始填充
    top, bottom, left, right = 0, n - 1, 0, n - 1  # 定义矩阵的边界

    while top <= bottom and left <= right:
        # 从左到右填充上边界
        for i in range(left, right + 1):
            matrix[top][i] = num
            num += 1
        top += 1  # 填充完上边界后，top边界向下移动

        # 从上到下填充右边界
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1  # 填充完右边界后，right边界向左移动

        # 确保当前边界仍然有效
        if top <= bottom:
            # 从右到左填充下边界
            for i in range(right, left - 1, -1):
                matrix[bottom][i] = num
                num += 1
            bottom -= 1  # 填充完下边界后，bottom边界向上移动

        # 确保当前边界仍然有效
        if left <= right:
            # 从下到上填充左边界
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1  # 填充完左边界后，left边界向右移动

    return matrix

```

### 步骤详解

1. **初始化矩阵**：

   ```python
   
   matrix = [[0] * n for _ in range(n)]
   ```

   这行代码创建了一个 n×nn \times nn×n 的矩阵，并用0填充。例如，当 n=3n = 3n=3 时，生成的矩阵是：

   ```
   [
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]
   ]
   ```

2. **初始化变量**：

   ```python
   num = 1
   top, bottom, left, right = 0, n - 1, 0, n - 1
   ```

   `num` 是当前要填入矩阵的数，初始值为1。`top`, `bottom`, `left`, `right` 分别表示矩阵的上、下、左、右边界。

3. **螺旋填充**：

   ```python
   
   while top <= bottom and left <= right:
   ```

   这个 `while` 循环确保在矩阵的边界内进行操作。

4. **填充上边界**：

   ```python
   for i in range(left, right + 1):
       matrix[top][i] = num
       num += 1
   top += 1
   ```

   这段代码从左到右填充上边界，然后将 `top` 边界向下移动一行。例如，在第一次迭代中填充的结果是：

   ```python
   [
     [1, 2, 3],
     [0, 0, 0],
     [0, 0, 0]
   ]
   ```

5. **填充右边界**：

   ```python
   for i in range(top, bottom + 1):
       matrix[i][right] = num
       num += 1
   right -= 1
   ```

   这段代码从上到下填充右边界，然后将 `right` 边界向左移动一列。第一次迭代后的结果是：

   ```python
   [
     [1, 2, 3],
     [0, 0, 4],
     [0, 0, 5]
   ]
   ```

6. **填充下边界**（在确保 `top <= bottom` 的情况下）：

   ```
   python复制代码if top <= bottom:
       for i in range(right, left - 1, -1):
           matrix[bottom][i] = num
           num += 1
       bottom -= 1
   ```

   这段代码从右到左填充下边界，然后将 `bottom` 边界向上移动一行。结果是：

   ```
   csharp复制代码[
     [1, 2, 3],
     [0, 0, 4],
     [7, 6, 5]
   ]
   ```

7. **填充左边界**（在确保 `left <= right` 的情况下）：

   ```python
   if left <= right:
       for i in range(bottom, top - 1, -1):
           matrix[i][left] = num
           num += 1
       left += 1
   ```

   这段代码从下到上填充左边界，然后将 `left` 边界向右移动一列。最终结果是：

   ```python
   [
     [1, 2, 3],
     [8, 9, 4],
     [7, 6, 5]
   ]
   ```

### 总结

通过不断调整矩阵的边界，我们实现了螺旋填充矩阵的效果。这种方法确保在遍历过程中不会重复或遗漏任何元素。每次遍历方向改变后，相应的边界都会收缩，直到所有元素都被填充到矩阵中。