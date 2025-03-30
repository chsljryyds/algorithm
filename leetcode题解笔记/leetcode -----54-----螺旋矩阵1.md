# leetcode -----54-----螺旋矩阵1

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

解法:

```python
def spiralOrder(matrix):
    if not matrix:
        return []

    # 定义边界
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    result = []

    while top <= bottom and left <= right:
        # 从左到右遍历上边界
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1

        # 从上到下遍历右边界
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        if top <= bottom:
            # 从右到左遍历下边界
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1

        if left <= right:
            # 从下到上遍历左边界
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result

# 示例用法
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(spiralOrder(matrix))  # 输出 [1, 2, 3, 6, 9, 8, 7, 4, 5]

```

### 题目分析

我们需要遍历一个矩阵，按照顺时针螺旋顺序输出矩阵中的所有元素。顺时针螺旋顺序意味着我们需要从矩阵的左上角开始，沿着外边界顺时针方向移动，逐层深入矩阵的内层，直到遍历完整个矩阵。

### 具体步骤

1. **定义边界**：初始化矩阵的四个边界，分别是上边界、下边界、左边界和右边界。

2. **遍历矩阵**：使用一个循环来遍历矩阵，按照顺时针螺旋顺序移动，并逐步缩小边界。

3. 移动方向

   ：按照右、下、左、上的顺序移动：

   - 从左到右遍历当前的上边界，然后上边界下移。
   - 从上到下遍历当前的右边界，然后右边界左移。
   - 从右到左遍历当前的下边界（如果还有未遍历的行），然后下边界上移。
   - 从下到上遍历当前的左边界（如果还有未遍历的列），然后左边界右移。

4. **终止条件**：当遍历范围超出边界时，终止循环。