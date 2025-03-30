# leetcode-------56 -------合并区间

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

 

解法：

```python
def merge(intervals):
    # 如果区间列表为空，直接返回空列表
    if not intervals:
        return []

    # 将区间按照起始位置进行排序
    intervals.sort(key=lambda x: x[0])

    # 初始化结果列表
    merged_intervals = []

    # 遍历每一个区间
    for interval in intervals:
        # 如果结果列表为空，或者当前区间与上一个区间不重叠，直接添加到结果列表
        if not merged_intervals or merged_intervals[-1][1] < interval[0]:
            merged_intervals.append(interval)
        else:
            # 如果当前区间与上一个区间重叠，更新上一个区间的结束位置
            merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])

    return merged_intervals

# 示例用法
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
merged_intervals = merge(intervals)
print(merged_intervals)  # 输出: [[1, 6], [8, 10], [15, 18]]

```

### 解释代码：

1. **排序**：

   ```python
   
   intervals.sort(key=lambda x: x[0])
   ```

   使用 `lambda x: x[0]` 作为排序的键函数，以每个区间的起始位置进行排序。

2. **初始化结果列表**：

   ```python
   
   merged_intervals = []
   ```

3. **遍历区间**：

   ```python
   
   for interval in intervals:
   ```

4. **判断是否合并**：

   - 如果结果列表为空，或者当前区间与上一个区间不重叠（即当前区间的起始位置大于上一个合并区间的结束位置），则直接添加当前区间到结果列表。
   - 如果当前区间与上一个区间重叠（即当前区间的起始位置小于等于上一个合并区间的结束位置），则更新上一个合并区间的结束位置为当前区间的结束位置和上一个合并区间结束位置中的最大值。

   ```python
   if not merged_intervals or merged_intervals[-1][1] < interval[0]:
       merged_intervals.append(interval)
   else:
       merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
   ```

通过以上步骤，我们可以合并所有重叠的区间，并返回一个不重叠的区间数组。



`merged_intervals[-1][1]` 是指列表 `merged_intervals` 中最后一个区间的结束位置。为了更详细地解释这个语法，让我们逐步拆解来看：

1. `merged_intervals` 是一个列表，用来存储合并后的区间。
2. `merged_intervals[-1]` 是列表中的最后一个元素。使用负索引 `-1` 可以访问列表中的最后一个元素，这是 Python 的一种特性。
3. `merged_intervals[-1][1]` 是最后一个区间的结束位置。每个区间本质上是一个包含两个元素的子列表，其中第一个元素（索引 0）是区间的起始位置，第二个元素（索引 1）是区间的结束位置。

例如，如果 `merged_intervals` 列表是 `[[1, 6], [8, 10]]`，那么：

- `merged_intervals[-1]` 就是 `[8, 10]`，即列表的最后一个区间。
- `merged_intervals[-1][1]` 就是 10，表示区间 `[8, 10]` 的结束位置。

在合并重叠区间的过程中，我们需要判断当前区间是否与最后一个合并后的区间重叠，并在必要时更新最后一个合并区间的结束位置。下面是代码段及其解释：

```python
if not merged_intervals or merged_intervals[-1][1] < interval[0]:
    merged_intervals.append(interval)
else:
    merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
```

### 解释：

- `if not merged_intervals or merged_intervals[-1][1] < interval[0]:`
  - 如果 `merged_intervals` 列表为空（即还没有任何合并后的区间），或者当前区间 `interval` 的起始位置 `interval[0]` 大于最后一个合并区间的结束位置 `merged_intervals[-1][1]`，这意味着当前区间 `interval` 与任何已合并的区间都不重叠，因此可以直接将当前区间添加到 `merged_intervals`。
- `else:`
  - 如果当前区间 `interval` 与最后一个合并区间重叠（即 `interval[0]` 小于或等于 `merged_intervals[-1][1]`），我们需要更新最后一个合并区间的结束位置 `merged_intervals[-1][1]`。新的结束位置应为当前区间 `interval` 的结束位置 `interval[1]` 与最后一个合并区间的结束位置 `merged_intervals[-1][1]` 中的较大值。

这段代码确保了所有重叠的区间被正确地合并，从而生成一个不重叠的区间列表。