# 976-----三角形的最大周长------leetcode

给定由一些正数（代表长度）组成的数组 `nums` ，返回 *由其中三个长度组成的、**面积不为零**的三角形的最大周长* 。如果不能形成任何面积不为零的三角形，返回 `0`。

 

```python
def largest_perimeter(nums):
    # Step 1: Sort the array in non-decreasing order
    nums.sort()
    
    # Step 2: Scan from the end to find the largest valid triplet (a, b, c)
    n = len(nums)
    max_perimeter = 0
    
    for i in range(n - 3, -1, -1):
        # Get the largest three elements available
        a = nums[i]
        b = nums[i + 1]
        c = nums[i + 2]
        
        # Check if they form a valid triangle
        if a + b > c:
            # Calculate the perimeter
            perimeter = a + b + c
            return perimeter
        
    # If no valid triangle found
    return 0

```

### 解题思路

为了构成一个三角形，任意三边长度需要满足三角形的性质：任意两边之和大于第三边。基于这个性质，我们可以通过以下步骤解决问题：

1. **排序数组**：首先将数组 nums 按照长度从小到大进行排序。
2. **扫描三元组**：从排序后的数组中，以从大到小的顺序取出三个相邻的长度（a, b, c）。这是因为这样取出的三个长度能够最大程度上满足三角形的边长性质。
3. **检查三角形条件**：对取出的三个长度 a, b, c，检查是否满足 a + b > c。如果满足，则这三个长度可以构成一个三角形，计算其周长并返回。
4. **返回结果**：如果遍历完整个数组后都找不到可以构成三角形的三个长度，则返回0。