# leetcode --- 删除指定值元素

## 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素。元素的顺序可能发生改变。然后返回 nums 中与 val 不同的元素的数量。 假设 nums 中不等于 val 的元素数量为 k，要通过此题，您需要执行以下操作： 更改 nums 数组，使 nums 的前 k 个元素包含不等于 val 的元素。nums 的其余元素和 nums 的大小并不重要。 返回 k。

解法

```python
def remove_element(nums, val):
    slow = 0
    
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
            
    return slow

# 示例用法
nums = [3, 2, 2, 3, 4, 2, 5]
val = 2
k = remove_element(nums, val)
print(k)  # 输出 3
print(nums[:k])  # 输出 [3, 3, 4, 5]

```

这道题同样可以使用双指针方法来实现。具体步骤如下：

1. **初始化两个指针**：一个慢指针 `slow` 指向数组的开头，用于记录不等于 `val` 的元素；一个快指针 `fast` 遍历整个数组。

2. 遍历数组

   ：使用快指针 

   ```
   fast
   ```

    遍历数组中的每个元素。

   - 如果 `nums[fast]` 不等于 `val`，将 `nums[fast]` 的值赋给 `nums[slow]`，然后将 `slow` 指针向前移动一位。

3. **返回结果**：遍历完成后，`slow` 指针的位置即为数组中不等于 `val` 的元素的个数，返回 `slow`。

这种方法只需遍历一次数组，时间复杂度为 O(n)，空间复杂度为 O(1)，非常高效。







### 详细解释：

1. **初始化**：

   ```
   slow = 0
   ```

   `slow` 指针初始化为 0，指向数组的开头。

2. **遍历数组**：

   ```
   for fast in range(len(nums)):
       if nums[fast] != val:
           nums[slow] = nums[fast]
           slow += 1
   ```

   - 从数组的第一个元素开始遍历（`fast` 从 0 开始）。
   - 如果 `nums[fast]` 不等于 `val`，将 `nums[fast]` 的值赋给 `nums[slow]`，然后将 `slow` 向前移动一位。

3. **返回结果**：

   ```
   
   return slow
   ```

   遍历结束后，`slow` 指针指向的即为数组中不等于 `val` 的元素的个数。

### 示例说明：

对于输入数组 `nums = [3, 2, 2, 3, 4, 2, 5]`，需要移除 `val = 2` 的元素，通过上述方法处理后：

- 初始状态：`slow = 0`, `fast = 0`
- `nums[0] == 3` 不等于 `val`，`nums[slow] = nums[fast]`，`slow` 增加 1。
- `nums[1] == 2` 等于 `val`，跳过。
- `nums[2] == 2` 等于 `val`，跳过。
- `nums[3] == 3` 不等于 `val`，`nums[slow] = nums[fast]`，`slow` 增加 1。
- 继续这个过程，最后 `nums` 变为 `[3, 3, 4, 5, ...]`（`...` 表示后面的元素不重要）。

最终返回 `k = 3`，表示前 3 个元素是 `[3, 3, 4]`，并且可以忽略数组其余部分。