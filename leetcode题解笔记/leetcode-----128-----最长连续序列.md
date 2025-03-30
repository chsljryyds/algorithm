# leetcode-----128-----最长连续序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。



解法：

```python
def longestConsecutive(nums):
    num_set = set(nums)  # 使用哈希表存储数组中的所有元素
    longest_streak = 0

    for num in num_set:
        # 只有当 num 是序列的起点时才处理
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            # 扩展序列
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            # 更新最长序列长度
            longest_streak = max(longest_streak, current_streak)

    return longest_streak

```



好的，让我们详细讲解一下这段代码的逻辑。

这段代码的目的是找到所有可能的连续序列，并计算每个序列的长度，然后记录最长的序列长度。关键部分是识别每个序列的起点，并从起点开始扩展序列。下面是逐步解释：

### 代码逻辑逐步解释

#### 1. `if num - 1 not in num_set:`

- 目的是判断当前数字 `num` 是否是某个连续序列的起点。
- 如果 `num - 1` 不在 `num_set` 中，那么 `num` 就是连续序列的起点。因为如果 `num` 不是序列的起点，那么在序列中必然存在一个比 `num` 小1的数（即 `num - 1`），这会导致从更小的数开始计算序列长度。

#### 2. `current_num = num` 和 `current_streak = 1`

- 初始化当前序列的起点 `current_num` 为 `num`，并且初始化当前序列的长度 `current_streak` 为 1。

#### 3. `while current_num + 1 in num_set:`

- 这个 `while` 循环的目的是扩展当前找到的连续序列。
- 检查 `current_num` 的下一个数字 `current_num + 1` 是否在集合 `num_set` 中。如果存在，则将 `current_num` 增加 1，并将 `current_streak` 增加 1。

### 举例说明

假设我们有一个数组 `nums = [100, 4, 200, 1, 3, 2]`，我们将详细说明这段代码如何找到最长的连续序列：

1. 将数组转换为集合：`num_set = {1, 2, 3, 4, 100, 200}`。
2. 遍历集合中的每个元素。

#### 遍历过程：

- **元素 1**：
  - `if 1 - 1 not in num_set:` 成立，因为 `0` 不在 `num_set` 中。
  - 初始化：`current_num = 1`, `current_streak = 1`。
  - 扩展序列：`1 + 1 = 2` 在 `num_set` 中，`current_num` 变为 `2`, `current_streak` 变为 `2`。
  - 扩展序列：`2 + 1 = 3` 在 `num_set` 中，`current_num` 变为 `3`, `current_streak` 变为 `3`。
  - 扩展序列：`3 + 1 = 4` 在 `num_set` 中，`current_num` 变为 `4`, `current_streak` 变为 `4`。
  - `4 + 1 = 5` 不在 `num_set` 中，停止扩展，记录最长序列长度为 `4`。
- **元素 2**：
  - `if 2 - 1 not in num_set:` 不成立，因为 `1` 在 `num_set` 中，跳过。
- **元素 3**：
  - `if 3 - 1 not in num_set:` 不成立，因为 `2` 在 `num_set` 中，跳过。
- **元素 4**：
  - `if 4 - 1 not in num_set:` 不成立，因为 `3` 在 `num_set` 中，跳过。
- **元素 100**：
  - `if 100 - 1 not in num_set:` 成立，因为 `99` 不在 `num_set` 中。
  - 初始化：`current_num = 100`, `current_streak = 1`。
  - `100 + 1 = 101` 不在 `num_set` 中，停止扩展，最长序列长度仍为 `4`。
- **元素 200**：
  - `if 200 - 1 not in num_set:` 成立，因为 `199` 不在 `num_set` 中。
  - 初始化：`current_num = 200`, `current_streak = 1`。
  - `200 + 1 = 201` 不在 `num_set` 中，停止扩展，最长序列长度仍为 `4`。

最终，最长的连续序列是 `[1, 2, 3, 4]`，长度为 `4`。