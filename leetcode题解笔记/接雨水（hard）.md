# **leetcode  题解笔记**(hard)



## 接雨水：

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。'''

`def trap(height):
    if not height:
        return 0
    
```python
left, right = 0, len(height) - 1
left_max, right_max = 0, 0
water_trapped = 0

while left < right:
    if height[left] < height[right]:
        if height[left] >= left_max:
            left_max = height[left]
        else:
            water_trapped += left_max - height[left]
        left += 1
    else:
        if height[right] >= right_max:
            right_max = height[right]
        else:
            water_trapped += right_max - height[right]
        right -= 1

return water_trapped
```

# 示例测试
height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(trap(height))  # 输出: 6`

### 详细解释

1. 初始化左指针 `left` 为 0，右指针 `right` 为数组的最后一个索引，即 `len(height) - 1`。

2. 初始化 `left_max` 和 `right_max` 为 0，分别表示左边和右边的最大高度。

3. 初始化 `water_trapped` 为 0，表示接到的雨水总量。

4. 进入 

   ```
   while
   ```

    循环，直到 

   ```
   left
   ```

    和 

   ```
   right
   ```

    相遇：

   - 如果 

     ```
     height[left] < height[right]
     ```

     ，则说明当前能接的雨水量取决于左边的高度：

     - 如果 `height[left] >= left_max`，更新 `left_max`；
     - 否则，累加 `left_max - height[left]` 到 `water_trapped`。
     - 移动左指针 `left` 向右。

   - 否则，当前能接的雨水量取决于右边的高度：

     - 如果 `height[right] >= right_max`，更新 `right_max`；
     - 否则，累加 `right_max - height[right]` 到 `water_trapped`。
     - 移动右指针 `right` 向左。

5. 最后返回 `water_trapped`。

这个方法的时间复杂度为 O(n)，空间复杂度为 O(1)，是一个高效的解决方案。



好的，我会详细解释代码是如何运行的，特别是当 `height = [0,1,0,2,1,0,1,3,2,1,2,1]` 时的具体过程。

### 初始状态

- `height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]`
- `left = 0`
- `right = 11`
- `left_max = 0`
- `right_max = 0`
- `water_trapped = 0`

### 详细步骤

#### 第一步

- `left = 0`, `right = 11`

- `height[left] = 0`, `height[right] = 1`

- 因为 

  ```
  height[left] < height[right]
  ```

  ，所以我们处理左边：

  - `height[left] = 0` < `left_max = 0`，更新 `left_max = 0`
  - 移动左指针：`left = 1`

#### 第二步

- `left = 1`, `right = 11`

- `height[left] = 1`, `height[right] = 1`

- 因为 

  ```
  height[left] >= height[right]
  ```

  ，所以我们处理右边：

  - `height[right] = 1` >= `right_max = 0`，更新 `right_max = 1`
  - 移动右指针：`right = 10`

#### 第三步

- `left = 1`, `right = 10`

- `height[left] = 1`, `height[right] = 2`

- 因为 

  ```
  height[left] < height[right]
  ```

  ，所以我们处理左边：

  - `height[left] = 1` >= `left_max = 0`，更新 `left_max = 1`
  - 移动左指针：`left = 2`

#### 第四步

- `left = 2`, `right = 10`

- `height[left] = 0`, `height[right] = 2`

- 因为 

  ```
  height[left] < height[right]
  ```

  ，所以我们处理左边：

  - `height[left] = 0` < `left_max = 1`，`water_trapped += left_max - height[left] = 1 - 0 = 1`
  - `water_trapped = 1`
  - 移动左指针：`left = 3`

#### 第五步

- `left = 3`, `right = 10`

- `height[left] = 2`, `height[right] = 2`

- 因为 

  ```
  height[left] >= height[right]
  ```

  ，所以我们处理右边：

  - `height[right] = 2` >= `right_max = 1`，更新 `right_max = 2`
  - 移动右指针：`right = 9`

#### 第六步

- `left = 3`, `right = 9`

- `height[left] = 2`, `height[right] = 1`

- 因为 

  ```
  height[left] >= height[right]
  ```

  ，所以我们处理右边：

  - `height[right] = 1` < `right_max = 2`，`water_trapped += right_max - height[right] = 2 - 1 = 1`
  - `water_trapped = 2`
  - 移动右指针：`right = 8`

#### 第七步

- `left = 3`, `right = 8`

- `height[left] = 2`, `height[right] = 2`

- 因为 

  ```
  height[left] >= height[right]
  ```

  ，所以我们处理右边：

  - `height[right] = 2` >= `right_max = 2`，更新 `right_max = 2`
  - 移动右指针：`right = 7`

#### 第八步

- `left = 3`, `right = 7`

- `height[left] = 2`, `height[right] = 3`

- 因为 

  ```
  height[left] < height[right]
  ```

  ，所以我们处理左边：

  - `height[left] = 2` >= `left_max = 1`，更新 `left_max = 2`
  - 移动左指针：`left = 4`

#### 第九步

- `left = 4`, `right = 7`

- `height[left] = 1`, `height[right] = 3`

- 因为 

  ```
  height[left] < height[right]
  ```

  ，所以我们处理左边：

  - `height[left] = 1` < `left_max = 2`，`water_trapped += left_max - height[left] = 2 - 1 = 1`
  - `water_trapped = 3`
  - 移动左指针：`left = 5`

#### 第十步

- `left = 5`, `right = 7`

- `height[left] = 0`, `height[right] = 3`

- 因为 

  ```
  height[left] < height[right]
  ```

  ，所以我们处理左边：

  - `height[left] = 0` < `left_max = 2`，`water_trapped += left_max - height[left] = 2 - 0 = 2`
  - `water_trapped = 5`
  - 移动左指针：`left = 6`

#### 第十一步

- `left = 6`, `right = 7`

- `height[left] = 1`, `height[right] = 3`

- 因为 

  ```
  height[left] < height[right]
  ```

  ，所以我们处理左边：

  - `height[left] = 1` < `left_max = 2`，`water_trapped += left_max - height[left] = 2 - 1 = 1`
  - `water_trapped = 6`
  - 移动左指针：`left = 7`

#### 结束条件

- `left = 7`, `right = 7`
- `left` 和 `right` 相遇，退出循环。

最终 `water_trapped = 6`，这就是最后接到的雨水量。

### 总结

在这个过程中，通过双指针法，从两端向中间移动，每次更新左边或右边的最大高度，计算当前高度与最大高度的差值，即为当前能接到的雨水量。这个方法高效且易于理解。