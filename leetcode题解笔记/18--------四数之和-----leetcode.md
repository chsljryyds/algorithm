# 18--------四数之和-----leetcode

给你一个由 `n` 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且**不重复**的四元组 `[nums[a], nums[b], nums[c], nums[d]]` （若两个四元组元素一一对应，则认为两个四元组重复）：

- `0 <= a, b, c, d < n`
- `a`、`b`、`c` 和 `d` **互不相同**
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

你可以按 **任意顺序** 返回答案 。



```python
def fourSum(nums, target):
    n = len(nums)
    if n < 4:
        return []
    
    nums.sort()
    result = []
    
    for i in range(n - 3):
        # skip duplicates
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n - 2):
            # skip duplicates
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            left = j + 1
            right = n - 1
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    # skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
    
    return result

```

这道题目是求解一个数组中，所有满足给定目标值的不重复四元组。我们可以利用类似于三数之和的方法来解决，但需要扩展到四个数的情况。

### 解题思路

1. **排序数组：** 首先对数组进行排序，这样可以利用双指针的方法来减少时间复杂度。
2. **双指针：** 使用两层循环固定两个数，然后使用双指针在剩余的数组部分中查找另外两个数，使它们的和等于目标值减去前两个数的和。
3. **去重处理：** 为了避免重复解，对于固定的第一个数和第二个数，需要跳过重复的情况。







当我们解决四数之和的问题时，我们需要考虑如何有效地找到所有满足条件的四元组 `[nums[a], nums[b], nums[c], nums[d]]`，使得它们的和等于给定的目标值 `target`。

### 1. 排序数组

首先，对输入的数组 `nums` 进行排序。排序有助于将重复元素聚集在一起，从而更容易处理重复的情况，并且利用有序数组可以更有效地利用双指针法。

```

nums.sort()
```

### 2. 双指针法

使用两层循环固定前两个数，并且在剩余的部分中使用双指针来寻找另外两个数，使它们的和等于 `target` 减去前两个数的和。

```
for i in range(n - 3):
    if i > 0 and nums[i] == nums[i - 1]:
        continue
    for j in range(i + 1, n - 2):
        if j > i + 1 and nums[j] == nums[j - 1]:
            continue
        left = j + 1
        right = n - 1
        while left < right:
            current_sum = nums[i] + nums[j] + nums[left] + nums[right]
            if current_sum == target:
                result.append([nums[i], nums[j], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
```

### 3. 去重处理

在双指针移动过程中，需要跳过重复的解。这是因为在排序后，如果当前元素与上一个元素相同，则可以跳过，避免产生重复的四元组。

- 外层循环中，如果 `nums[i]` 与前一个元素相同，则跳过，避免重复。
- 内层循环中，如果 `nums[j]` 与前一个元素相同，则跳过，避免重复。
- 在找到一个满足条件的四元组后，还需要在移动 `left` 和 `right` 指针时，跳过连续相同的元素。

### 4. 结果收集

每当找到一个满足条件的四元组时，将其加入到结果列表 `result` 中。

### 代码详解

#### 外层循环

```
for i in range(n - 3):
    if i > 0 and nums[i] == nums[i - 1]:
        continue
```

- `for i in range(n - 3):` 遍历数组，固定第一个数 `nums[i]`。
- `if i > 0 and nums[i] == nums[i - 1]:` 如果 `nums[i]` 与前一个元素相同，则跳过，避免重复解。

#### 内层循环

```
for j in range(i + 1, n - 2):
    if j > i + 1 and nums[j] == nums[j - 1]:
        continue
```

- `for j in range(i + 1, n - 2):` 在固定第一个数后，遍历数组，固定第二个数 `nums[j]`。
- `if j > i + 1 and nums[j] == nums[j - 1]:` 如果 `nums[j]` 与前一个元素相同，则跳过，避免重复解。

#### 双指针查找

```
left = j + 1
right = n - 1
while left < right:
    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
    if current_sum == target:
        result.append([nums[i], nums[j], nums[left], nums[right]])
        while left < right and nums[left] == nums[left + 1]:
            left += 1
        while left < right and nums[right] == nums[right - 1]:
            right -= 1
        left += 1
        right -= 1
    elif current_sum < target:
        left += 1
    else:
        right -= 1
```

- `left` 和 `right` 分别指向剩余数组的两端。
- `while left < right:` 循环进行双指针查找。
- `current_sum = nums[i] + nums[j] + nums[left] + nums[right]` 计算当前四个数的和。
- 如果 `current_sum == target`，则找到一个符合条件的四元组，加入到 `result` 中，并且在移动 `left` 和 `right` 指针时，跳过重复的元素。
- 如果 `current_sum < target`，则增加 `left` 指针以增加和的值。
- 如果 `current_sum > target`，则减小 `right` 指针以减小和的值。

### 总结

通过以上详细讲解，我们了解了如何利用排序和双指针法解决四数之和的问题。这种方法有效地减少了时间复杂度，使得算法能够在合理的时间内找到所有满足条件的四元组。