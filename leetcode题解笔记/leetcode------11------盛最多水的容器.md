# leetcode------11------盛最多水的容器

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。



解法：

```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        # 计算宽度和高度
        width = right - left
        current_height = min(height[left], height[right])
        
        # 计算当前容量
        current_area = width * current_height
        
        # 更新最大容量
        max_area = max(max_area, current_area)
        
        # 移动较短垂线的指针
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

# 测试
height = [1,8,6,2,5,4,8,3,7]
print(max_area(height))  # 输出49

```



这个问题是一个经典的双指针问题，目的是找出一个数组中两个位置的垂线，与 x 轴共同构成的容器可以容纳最多的水。让我们一步步详细讲解如何解决这个问题。

### 问题理解

首先，我们需要理解问题的图形表示：

- 数组 `height` 表示的是每个位置上垂线的高度。
- 我们要找到两个位置 `i` 和 `j` (其中 `i < j`)，使得它们之间的区域可以容纳最多的水。

水的容量由以下两个因素决定：

1. 宽度：两个垂线之间的距离，即 `j - i`。
2. 高度：两个垂线中较短的那一条，即 `min(height[i], height[j])`。

水的容量公式为： 容量=宽度×高度=(j−i)×min⁡(height[i],height[j])\text{容量} = \text{宽度} \times \text{高度} = (j - i) \times \min(height[i], height[j])容量=宽度×高度=(j−i)×min(height[i],height[j])

### 解法：双指针法

为了找到这两个垂线，我们可以使用双指针法。双指针法的核心思想是：

1. 初始化两个指针，一个指向数组的开头（left），一个指向数组的末尾（right）。
2. 计算这两个指针之间的水容量。
3. 移动较短的那一条垂线的指针，以期找到更高的垂线，从而可能增加容量。
4. 重复上述过程，直到两个指针相遇。

### 具体步骤

1. 初始化 `left` 指针为数组开头，`right` 指针为数组末尾，最大水容量 `max_area` 初始化为 0。

2. 在 

   ```
   left
   ```

    小于 

   ```
   right
   ```

    的情况下，重复以下步骤：

   - 计算当前容器的宽度：`width = right - left`
   - 计算当前容器的高度：`height = min(height[left], height[right])`
   - 计算当前容器的容量：`current_area = width * height`
   - 更新最大水容量：`max_area = max(max_area, current_area)`
   - 移动较短垂线的指针。如果 `height[left] < height[right]`，则 `left += 1`；否则，`right -= 1`

3. 当 `left` 与 `right` 相遇时，返回 `max_area`。





### 解释代码

1. **初始化指针和变量**：`left` 从左边开始，`right` 从右边开始，`max_area` 用来记录最大容量。
2. **循环计算**：当 `left` 小于 `right` 时，计算当前容器的宽度和高度，然后计算容量并更新最大容量。
3. **移动指针**：根据高度比较结果，移动较短垂线的指针，以期找到更高的垂线，从而可能增加容量。
4. **返回结果**：当循环结束时，返回最大容量。

这样，通过双指针法，我们可以高效地找到能够容纳最多水的两个垂线。这个方法的时间复杂度是 O(n)，非常高效。



