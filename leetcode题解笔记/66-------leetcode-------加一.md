# 66-------leetcode-------加一

给定一个由 **整数** 组成的 **非空** 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储**单个**数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。



解法：

```python
def plusOne(digits):
    n = len(digits)
    
    # 从数组的最后一位开始加一
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        else:
            digits[i] = 0
    
    # 如果最高位也有进位，需要在数组最前面插入一个1
    return [1] + digits

```

### 解题思路

1. **理解题意**：
   - 数组中的每个元素是一个数字的一位，整个数组按照从高位到低位表示一个非负整数。
   - 要求对这个整数加一，返回加一后的结果。
2. **加一操作的考虑**：
   - 从数组的末尾开始遍历，将当前元素加一。
   - 如果加一后不产生进位（即不需要进位），则直接返回数组。
   - 如果产生进位，将当前位置的数字置为0，并继续向前进位。
3. **特殊情况处理**：
   - 如果最高位产生了进位，例如999 + 1 = 1000，需要在数组最前面插入一个1。
4. **算法实现**：
   - 从数组的最后一个元素开始向前遍历。
   - 加一操作并判断是否需要进位。
   - 最后如果遍历完成还有进位，则在数组的最前面插入一个1。