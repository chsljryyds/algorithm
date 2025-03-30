# 146LCR-------leetcode-------螺旋遍历二维数组

给定一个二维数组 `array`，请返回「**螺旋遍历**」该数组的结果。

**螺旋遍历**：从左上角开始，按照 **向右**、**向下**、**向左**、**向上** 的顺序 **依次** 提取元素，然后再进入内部一层重复相同的步骤，直到提取完所有元素。



```python
def spiralOrder(matrix):
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse from left to right
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1

        # Traverse from top to bottom
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        if top <= bottom:
            # Traverse from right to left
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1

        if left <= right:
            # Traverse from bottom to top
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result

```

螺旋遍历二维数组是一个常见的算法问题，下面我来详细解释如何实现这个功能。

### 思路分析

螺旋遍历的过程可以描述为：

1. 从左上角开始，按照向右、向下、向左、向上的顺序依次遍历元素。
2. 每完成一轮遍历，缩小可以遍历的范围，继续按照相同的顺序进行下一轮遍历，直到遍历完所有元素。

### 实现步骤

1. **初始化边界指针**：
   - `top`：当前可以遍历的最上边界
   - `bottom`：当前可以遍历的最下边界
   - `left`：当前可以遍历的最左边界
   - `right`：当前可以遍历的最右边界
   - 这些指针初始时分别指向数组的边界。
2. **循环遍历**：
   - 按照向右、向下、向左、向上的顺序遍历数组，同时更新对应的边界指针。
   - 每次遍历完一个方向后，检查是否需要继续遍历（即判断边界指针是否仍然满足遍历条件）。
3. **整理结果**：
   - 将遍历过程中收集到的元素整理成结果数组。



### 解释实现细节

- **初始化**：首先初始化四个边界指针 `top, bottom, left, right`，它们分别代表当前可以遍历的上下左右边界。
- **循环遍历**：按照从左到右、从上到下、从右到左、从下到上的顺序遍历数组，并在每个方向上更新相应的边界指针。
- **结果整理**：将每次遍历得到的元素添加到结果列表 `result` 中。

这样，通过以上算法，可以有效地完成二维数组的螺旋遍历，时间复杂度为 O(m*n)，其中 m 和 n 分别为二维数组的行数和列数。