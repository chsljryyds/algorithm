# leetcode ------删除重复重现的元素

## 给你一个 非严格递增排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。然后返回 nums 中唯一元素的个数。 考虑 nums 的唯一元素的数量为 k ，你需要做以下事情确保你的题解可以被通过： 更改数组 nums ，使 nums 的前 k 个元素包含唯一元素，并按照它们最初在 nums 中出现的顺序排列。nums 的其余元素与 nums 的大小不重要。 返回 k 。

解法：


    

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    
    slow = 0
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1

# 示例用法
nums = [1, 1, 2, 2, 3, 4, 4]
k = remove_duplicates(nums)
print(k)  # 输出 4
print(nums[:k])  # 输出 [1, 2, 3, 4]

```

# 示例用法
nums = [1, 1, 2, 2, 3, 4, 4]
k = remove_duplicates(nums)
print(k)  # 输出 4
print(nums[:k])  # 输出 [1, 2, 3, 4]





1. **初始化两个指针**：一个慢指针 `slow` 指向数组的第一个元素，用于存放唯一元素；一个快指针 `fast` 遍历整个数组。

2. 遍历数组

   ：使用快指针 

   ```
   fast
   ```

    遍历数组中的每个元素。

   - 如果 `nums[fast]` 不等于 `nums[slow]`，说明遇到了一个新的唯一元素，将 `slow` 指针向前移动一位，并将 `nums[fast]` 的值赋给 `nums[slow]`。

3. **返回结果**：遍历完成后，`slow` 指针的位置即为数组中唯一元素的个数，返回 `slow + 1`。

这种方法只需要遍历一次数组，时间复杂度为 O(n)，空间复杂度为 O(1)，非常高效。







### 详细解释：

1. **初始化**：

   ```
   
   slow = 0
   ```

   `slow` 指针初始化为 0，指向第一个元素。

2. **遍历数组**：

   ```
   for fast in range(1, len(nums)):
       if nums[fast] != nums[slow]:
           slow += 1
           nums[slow] = nums[fast]
   ```

   - 从第二个元素开始遍历（`fast` 从 1 开始）。
   - 如果 `nums[fast]` 和 `nums[slow]` 不同，说明遇到新的唯一元素，将 `slow` 向前移动一位，并将 `nums[fast]` 赋给 `nums[slow]`。

3. **返回结果**：

   ```
   
   return slow + 1
   ```

   遍历结束后，`slow` 指针指向最后一个唯一元素的位置，因此唯一元素的数量为 `slow + 1`。

### 示例说明：

对于输入数组 `nums = [1, 1, 2, 2, 3, 4, 4]`，通过上述方法处理后：

- 初始状态：`slow = 0`, `fast = 1`
- `nums[1] == nums[0]`，跳过。
- `nums[2] != nums[0]`，`slow` 增加 1，`nums[1]` 设为 `nums[2]`，`nums` 变为 `[1, 2, 2, 2, 3, 4, 4]`。
- 继续这个过程，最后 `nums` 变为 `[1, 2, 3, 4, ...]`（`...` 表示后面的元素不重要）。

最终返回 `k = 4`，表示前 4 个元素是唯一的 `[1, 2, 3, 4]`。