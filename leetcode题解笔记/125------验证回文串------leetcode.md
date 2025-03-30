# 125------验证回文串------leetcode

如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，短语正着读和反着读都一样。则可以认为该短语是一个 **回文串** 。

字母和数字都属于字母数字字符。

给你一个字符串 `s`，如果它是 **回文串** ，返回 `true` ；否则，返回 `false` 。

 

```python
def isPalindrome(s: str) -> bool:
    # Step 1: Convert to lowercase and remove non-alphanumeric characters
    normalized_str = ''.join(c.lower() for c in s if c.isalnum())
    
    # Step 2: Check if the normalized string is a palindrome
    return normalized_str == normalized_str[::-1]

# Example usage:
s1 = "A man, a plan, a canal: Panama"
s2 = "race a car"

print(isPalindrome(s1))  # Output: True
print(isPalindrome(s2))  # Output: False

```

当解决这道题时，我们需要遵循以下步骤：

1. **转换为小写字符并移除非字母数字字符：**
   - 将字符串中的所有大写字母转换为小写字母。
   - 移除所有非字母数字字符，只保留字母和数字。
2. **判断是否为回文串：**
   - 回文串是指正着读和反着读都一样的字符串。

### 具体步骤和代码实现：

1. **转换为小写字符并移除非字母数字字符**：
   - 使用Python的字符串方法和列表推导来完成。
   - `isalnum()` 方法可以检查一个字符串是否只包含字母和数字。
2. **判断是否为回文串**：
   - 对处理后的字符串进行正向和反向比较即可。

下面是完整的Python代码实现：



当解决这道题时，我们需要遵循以下步骤：

1. **转换为小写字符并移除非字母数字字符：**
   - 将字符串中的所有大写字母转换为小写字母。
   - 移除所有非字母数字字符，只保留字母和数字。
2. **判断是否为回文串：**
   - 回文串是指正着读和反着读都一样的字符串。

### 具体步骤和代码实现：

1. **转换为小写字符并移除非字母数字字符**：
   - 使用Python的字符串方法和列表推导来完成。
   - `isalnum()` 方法可以检查一个字符串是否只包含字母和数字。
2. **判断是否为回文串**：
   - 对处理后的字符串进行正向和反向比较即可。





### 解释代码：

- `isPalindrome` 函数接受一个字符串 `s`，并返回一个布尔值，表示是否是回文串。
- `normalized_str` 使用了列表推导式，将 `s` 中的每个字符转换为小写（`c.lower()`），并且只保留字母和数字字符（`if c.isalnum()`）。
- `normalized_str[::-1]` 则是 `normalized_str` 的反向字符串。
- 最后，函数返回比较 `normalized_str` 和它的反向字符串是否相等的结果。

这样，我们就完成了对给定字符串是否为回文串的判断。







