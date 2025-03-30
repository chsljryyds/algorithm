# 179-------最大数------leetcode

给定一组非负整数 `nums`，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

**注意：**输出结果可能非常大，所以你需要返回一个字符串而不是整数。

 

**示例 1：**

```
输入：nums = [10,2]
输出："210"
```

**示例 2：**

```
输入：nums = [3,30,34,5,9]
输出："9534330"
```



```python
from typing import List

class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # Convert all numbers to strings
        nums = list(map(str, nums))
        
        # Custom comparator function for sorting
        def compare(x: str, y: str) -> int:
            if x + y > y + x:
                return -1  # x should come before y
            elif x + y < y + x:
                return 1   # y should come before x
            else:
                return 0
        
        # Sort using the custom comparator
        nums.sort(key=functools.cmp_to_key(compare))
        
        # Join sorted numbers to form the largest number
        largest_number = ''.join(nums)
        
        # Remove leading zeros if any
        largest_number = largest_number.lstrip('0')
        
        # If the entire number is zeros, return '0'
        if not largest_number:
            return '0'
        
        return largest_number

# Example usage:
nums1 = [10, 2]
nums2 = [3, 30, 34, 5, 9]

sol = Solution()
print(sol.largestNumber(nums1))  # Output: "210"
print(sol.largestNumber(nums2))  # Output: "9534330"


```

要解决这个问题，我们需要重新排列给定数组中的数字，使得它们拼接起来形成的数字最大。这种排序不是简单的数值大小比较，而是根据自定义规则进行的。

### 解题思路

1. **自定义排序规则**：
   - 对于两个数字 aaa 和 bbb，我们要确定它们在最终拼接结果中的顺序。
   - 如果将 aaa 和 bbb 转换为字符串后， a+ba + ba+b 大于 b+ab + ab+a，则 aaa 应该排在 bbb 前面，否则 bbb 应该排在 aaa 前面。
2. **实现步骤**：
   - 将数组中的所有数字转换为字符串。
   - 使用自定义的比较函数进行排序。
   - 将排序后的字符串数组连接起来形成最终的结果。
3. **特殊情况处理**：
   - 如果数组中的所有数字都是0，则最终结果应该是单个字符 "0"。





### 解释代码：

- `map(str, nums)`: 将数组中的每个整数转换为字符串。
- `compare` 函数：自定义比较函数，根据题目要求确定两个字符串的排列顺序。
- `functools.cmp_to_key(compare)`: 将比较函数转换为排序关键字。
- `nums.sort(key=functools.cmp_to_key(compare))`: 使用自定义的排序关键字对数组进行排序。
- `''.join(nums)`: 将排好序的字符串数组连接成一个字符串。
- `largest_number.lstrip('0')`: 去除最终结果中可能存在的前导零。
- 最后，如果最终结果是空字符串（即全部是零），则返回单个字符 "0"。

这样就能够得到按题目要求拼接后的最大整数的字符串表示。





4o

fuctools.cmp_to_key是什么



ChatGPT

`functools.cmp_to_key` 是 Python 标准库 `functools` 中的一个函数，用于将老式的比较函数（即接受两个参数并返回一个负数、零或正数的函数）转换为一个 key 函数，以便在排序算法中使用。

在 Python 2.x 中，比较函数是一种可以直接传递给排序函数（如 `list.sort()` 或 `sorted()`）的函数，其返回值决定了元素的排序顺序。然而，Python 3.x 中这种直接传递比较函数的方式已经被移除，取而代之的是使用 `key` 参数，该参数接受一个函数，该函数接受单个参数并返回一个用于排序的键。为了兼容老式的比较函数，Python 提供了 `functools.cmp_to_key` 函数。

### 用法示例：

假设有一个老式的比较函数 `compare(a, b)`，它返回负数、零或正数，表示 `a` 应该排在 `b` 之前、相同位置或之后。在 Python 3.x 中，如果想使用这个比较函数来对列表进行排序，可以通过 `functools.cmp_to_key` 将其转换为一个 key 函数，如下所示：

```python
import functools

# 定义老式的比较函数
def compare(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0

# 转换为 key 函数
key_func = functools.cmp_to_key(compare)

# 使用 key 函数进行排序
nums = [3, 1, 4, 1, 5, 9]
sorted_nums = sorted(nums, key=key_func)

print(sorted_nums)  # Output: [1, 1, 3, 4, 5, 9]
```

在这个例子中，`compare` 是一个简单的比较函数，它根据数字的大小返回负数、零或正数。通过 `functools.cmp_to_key` 将其转换为 `key_func`，然后可以将 `key_func` 作为 `sorted()` 函数的 `key` 参数来对 `nums` 进行排序，得到按照 `compare` 函数定义的顺序排列的列表。

总结来说，`functools.cmp_to_key` 可以让我们在 Python 3.x 中仍然能够使用老式的比较函数来进行排序操作。