# 389----- ---找不同----leetcode

给定两个字符串 `s` 和 `t` ，它们只包含小写字母。

字符串 `t` 由字符串 `s` 随机重排，然后在随机位置添加一个字母。

请找出在 `t` 中被添加的字母。

 

**示例 1：**

```
输入：s = "abcd", t = "abcde"
输出："e"
解释：'e' 是那个被添加的字母。
```

**示例 2：**

```
输入：s = "", t = "y"
输出："y"
```





```python
def find_extra_char(s, t):
    # Initialize an array to count occurrences of each character
    s_count = [0] * 26
    t_count = [0] * 26
    
    # Count occurrences in string s
    for char in s:
        s_count[ord(char) - ord('a')] += 1
    
    # Update counts based on string t
    for char in t:
        t_count[ord(char) - ord('a')] += 1
    
    # Find the extra character
    for i in range(26):
        if s_count[i] != t_count[i]:
            return chr(i + ord('a'))
    return ''

# Example usage:
s = "abcd"
t = "abcde"
extra_char = find_extra_char(s, t)
print("The extra character added in t is:", extra_char)



```

这道题目要求我们找出字符串 t 中添加的那个额外的字符，它是通过随机重排字符串 s 并在随机位置添加一个字母形成的。

我们可以利用字符的 ASCII 码进行解题。具体步骤如下：

1. **计数字符出现次数**: 首先，我们需要统计字符串 s 中每个字符出现的次数，可以利用一个大小为 26 的数组来记录，数组下标表示字符 'a' 到 'z'，数组的值表示对应字符出现的次数。
2. **遍历字符串 t**: 然后，遍历字符串 t 并更新计数数组。对于 t 中的每个字符，我们在计数数组中将对应位置的计数减去 1。
3. **找出添加的字符**: 最后，查找计数数组中值为 1 的位置，即为在 t 中被添加的字符。







### 解释代码：

- **计数数组 `count`**: 使用一个长度为 26 的数组来统计每个字符出现的次数。
- **统计字符串 s**: 遍历字符串 s，将每个字符的计数加一。
- **更新字符串 t**: 遍历字符串 t，将每个字符的计数减一。
- **找出额外字符**: 遍历计数数组，找到值为 1 的位置，并通过 `chr(i + ord('a'))` 找到对应的字符。

这种方法的时间复杂度为 O(n)，其中 n 是字符串 t 的长度，因为我们只需要线性地遍历两个字符串和计数数组。





4o

ord是什么



ChatGPT

在Python中，`ord()` 是一个内置函数，它用于返回字符的 ASCII 数值或 Unicode 数值（如果字符是 Unicode 的话）。具体来说：

- 对于 ASCII 字符集（包括大多数常见的英文字符），`ord()` 返回对应字符的 ASCII 数值。
- 对于 Unicode 字符，`ord()` 返回该字符的 Unicode 码点。

例如，`ord('a')` 返回的是整数 97，因为小写字母 'a' 的 ASCII 码值为 97。同样地，`ord('A')` 返回的是整数 65，因为大写字母 'A' 的 ASCII 码值为 65。

在上面的代码中，我们使用 `ord(char)` 来获取字符 `char` 的 ASCII 码值，并通过 `ord(char) - ord('a')` 将其转换为在计数数组中的索引（从 0 到 25），这样我们可以在数组中进行统计和访问。







ord(char) - ord('a')  这个是什么意思





`ord(char) - ord('a')` 这个表达式的意思是将字符 `char` 转换成以字母 'a' 为基准的偏移量。

让我详细解释一下：

1. **`ord('a')`**: 这是字符 'a' 的 ASCII 码值，即 97。
2. **`ord(char)`**: 这是字符 `char` 的 ASCII 码值。例如，如果 `char` 是 'a'，那么 `ord('a')` 就是 97；如果 `char` 是 'b'，那么 `ord('b')` 就是 98，依此类推。
3. **`ord(char) - ord('a')`**: 这个表达式计算了字符 `char` 相对于字符 'a' 的偏移量。例如：
   - 如果 `char` 是 'a'，则 `ord('a') - ord('a')` 等于 0。
   - 如果 `char` 是 'b'，则 `ord('b') - ord('a')` 等于 1。
   - 如果 `char` 是 'z'，则 `ord('z') - ord('a')` 等于 25。

在上面的代码中，我们使用这个表达式来将字符 `char` 映射到一个以字母 'a' 为起点的索引位置，这样就可以在长度为 26 的计数数组中进行统计和访问。