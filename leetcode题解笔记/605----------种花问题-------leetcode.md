# 605----------种花问题-------leetcode

假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。

给你一个整数数组 `flowerbed` 表示花坛，由若干 `0` 和 `1` 组成，其中 `0` 表示没种植花，`1` 表示种植了花。另有一个数 `n` ，能否在不打破种植规则的情况下种入 `n` 朵花？能则返回 `true` ，不能则返回 `false` 。



```python
def can_place_flowers(flowerbed, n):
    count = 0
    i = 0
    while i < len(flowerbed):
        if flowerbed[i] == 0 and (i == 0 or flowerbed[i - 1] == 0) and (i == len(flowerbed) - 1 or flowerbed[i + 1] == 0):
            flowerbed[i] = 1  # 在当前位置种花
            count += 1
        if count >= n:
            return True
        i += 1
    return False

# 示例用法
flowerbed1 = [1, 0, 0, 0, 1]
n1 = 1
print(can_place_flowers(flowerbed1, n1))  # 输出: True

flowerbed2 = [1, 0, 0, 0, 1]
n2 = 2
print(can_place_flowers(flowerbed2, n2))  # 输出: False

```

当解决这个问题时，我们需要在不打破相邻花朵规则的情况下，向花坛中种入 n 朵花。我们可以使用贪心算法来解决这个问题。

以下是用 Python 实现的解决方案：



### 解释：

1. **循环遍历花坛：** 我们从左到右遍历花坛中的每个位置。
2. **判断种花条件：** 对于每个位置 `i`，我们检查它是否为0（即没有种植花）且它左右相邻位置也为0（或者是边界情况），如果满足则可以在这个位置种花。
3. **种花并计数：** 如果可以在当前位置种花，则在 `flowerbed` 中将该位置置为1，并增加计数器 `count`。
4. **判断是否达到要求：** 每次种花后都检查计数器 `count` 是否达到了要种植的花的数量 `n`，如果达到则返回 `True`。
5. **循环结束：** 如果遍历完整个花坛都没有达到要求的花的数量，最终返回 `False`。

这种方法是有效的，因为在每次种花时我们总是尽量选择能种花的位置，以达到最优解。





如果直接遍历并尝试种花的方法导致超出时间限制，我们可以优化算法以减少不必要的遍历次数。下面是一个优化后的解决方案：

```python
def can_place_flowers(flowerbed, n):
    count = 0
    i = 0
    while i < len(flowerbed):
        if flowerbed[i] == 0 and (i == 0 or flowerbed[i - 1] == 0) and (i == len(flowerbed) - 1 or flowerbed[i + 1] == 0):
            flowerbed[i] = 1  # 在当前位置种花
            count += 1
        if count >= n:
            return True
        # 跳到下一个可种花的位置
        i += 2 if flowerbed[i] == 1 else 1
    
    return False

# 示例用法
flowerbed1 = [1, 0, 0, 0, 1]
n1 = 1
print(can_place_flowers(flowerbed1, n1))  # 输出: True

flowerbed2 = [1, 0, 0, 0, 1]
n2 = 2
print(can_place_flowers(flowerbed2, n2))  # 输出: False
```

### 优化解释：

1. **跳步遍历：** 在遍历过程中，我们可以直接跳过已经种植花的位置（即 `flowerbed[i] == 1`），因为相邻的位置不能种花。
2. **遍历逻辑：** 如果当前位置 `i` 可以种花（满足条件），我们在此位置种花，并将计数器 `count` 增加。然后我们跳到下一个可能的种花位置，即 `i += 2`（因为我们跳过了当前位置和下一个位置）或 `i += 1`（如果当前位置不能种花）。
3. **达到数量条件：** 每次种花后都检查计数器 `count` 是否达到了要求的花的数量 `n`，如果达到则返回 `True`。
4. **循环结束：** 如果遍历完整个花坛都没有达到要求的花的数量，最终返回 `False`。

这种优化方法减少了无效的遍历次数，通过跳步遍历可以更快地找到可种花的位置，从而在更短的时间内完成判断。