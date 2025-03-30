# leetcode------15-------三数之和

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请

你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。



解法：

```python
def three_sum(nums):
    nums.sort()  # 排序数组
    result = []
    
    for i in range(len(nums) - 2):
        # 跳过重复元素
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # 移动左指针，跳过重复元素
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # 移动右指针，跳过重复元素
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif sum < 0:
                left += 1
            else:
                right -= 1
    
    return result

# 示例测试
nums = [-1, 0, 1, 2, -1, -4]
print(three_sum(nums))

```

这是一个经典的三数之和问题。我们可以通过排序和双指针法来高效地解决这个问题。具体步骤如下：

1. **排序数组**：首先将数组进行排序，这样可以方便我们使用双指针法。
2. **遍历数组**：使用一个指针遍历数组，假设当前指针为 `i`，然后在 `i` 之后的子数组中使用双指针来寻找另外两个数。
3. **使用双指针**：
   - 初始化两个指针：左指针 `left` 放在 `i+1`，右指针 `right` 放在数组的末尾。
   - 计算三个数的和：
     - 如果和为零，记录这个三元组，并且移动两个指针，避免重复三元组。
     - 如果和小于零，说明需要更大的数，因此左指针右移。
     - 如果和大于零，说明需要更小的数，因此右指针左移。
4. **避免重复**：在移动指针和遍历时需要跳过重复的元素，以确保不出现重复的三元组。





在这个问题中，我们需要找到三个数的组合，使得它们的和为零。使用双指针法时，我们需要确保有足够的空间来放置三个指针（即 `i`, `left`, 和 `right`）。

假设数组长度为 `n`，指针 `i` 的范围需要是从 `0` 到 `n-3`，因为：

- `left` 指针是 `i + 1`，它需要至少有一个元素，所以 `i` 最多能到 `n-3`。
- `right` 指针是数组末尾，即 `n-1`。

这样就保证了 `i`、`left` 和 `right` 指针都有足够的空间，因此循环条件是 `i` 在 `[0, n-3]` 范围内遍历。具体代码如下：