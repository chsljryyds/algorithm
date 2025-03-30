# leetcode ----- 两数之和

## easy

### 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。 你可以按任意顺序返回答案。



### 解法：

```python
def two_sum(nums, target):
    # 创建一个哈希表用于存储数值和下标
    hash_map = {}
    # 遍历数组
    for i, num in enumerate(nums):
        # 计算当前数值与目标值的差值
        complement = target - num
        # 检查哈希表中是否存在这个差值
        if complement in hash_map:
            # 如果存在，返回这两个数的下标
            return [hash_map[complement], i]
        # 将当前数值和下标存入哈希表中
        hash_map[num] = i
    # 如果没有找到符合条件的两个数，返回空列表
    return []

# 示例用法
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)  # 输出 [0, 1]

```



# 示例用法
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)  # 输出 [0, 1]



### 详细流程

1. **初始化哈希表**：创建一个空的哈希表 `hash_map`。

2. 遍历数组

   ：

   - 对于数组中的每一个元素 `num`，计算它与目标值 `target` 的差值 `complement = target - num`。

   - 检查 

     ```
     complement
     ```

      是否存在于 

     ```
     hash_map
     ```

      中：

     - 如果存在，说明我们找到了两个数，它们的和等于 `target`，返回这两个数的下标。
     - 如果不存在，将当前数 `num` 和它的下标 `i` 存入 `hash_map`。

3. **返回结果**：如果遍历完数组后仍未找到符合条件的数对，则返回一个空列表。

### 举个例子

假设输入数组为 `nums = [2, 7, 11, 15]`，目标值为 `target = 9`。

- 初始化 `hash_map` 为 `{}`。
- 遍历第一个元素 `2`，计算 `complement = 9 - 2 = 7`，`7` 不在 `hash_map` 中，将 `2` 和它的下标 `0` 存入 `hash_map`，`hash_map` 变为 `{2: 0}`。
- 遍历第二个元素 `7`，计算 `complement = 9 - 7 = 2`，`2` 在 `hash_map` 中，找到符合条件的数对，返回 `[0, 1]`。

这样，我们用哈希表在一次遍历中就找到了答案，时间复杂度为 O(n)。