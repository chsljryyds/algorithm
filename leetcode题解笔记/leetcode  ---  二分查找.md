#   leetcode  ---  二分查找

## 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。



### 解法：

```python
def binary_search(nums, target):
left, right = 0, len(nums) - 1

while left <= right:
    mid = left + (right - left) // 2
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        left = mid + 1
    else:
        right = mid - 1

return -1
```

# 示例测试
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target = 7
print(binary_search(nums, target))  # 输出: 6

target = 11
print(binary_search(nums, target))  # 输出: -1





### 二分查找算法思路

1. 初始化两个指针：`left` 指向数组的起始位置，`right` 指向数组的末尾位置。

2. 进入循环，条件是 

   ```
   left
   ```

    小于等于 

   ```
   right
   ```

   ：

   - 计算中间位置 `mid`，即 `mid = left + (right - left) // 2`。
   - 如果 `nums[mid]` 等于目标值 `target`，返回 `mid`。
   - 如果 `nums[mid]` 小于 `target`，说明目标值在右半部分，更新 `left = mid + 1`。
   - 如果 `nums[mid]` 大于 `target`，说明目标值在左半部分，更新 `right = mid - 1`。

3. 如果循环结束时没有找到目标值，返回 -1。