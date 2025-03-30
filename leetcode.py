'''
124 .   二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。
'''


class Solution:
    def maxPathSum(self, root):
        self.max_sum = float('-inf')

        def max_gain(node):
            if not node:
                return 0

            # 递归计算左右子树的最大贡献值
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # 计算当前节点的最大路径和
            price_newpath = node.val + left_gain + right_gain

            # 更新全局最大路径和
            self.max_sum = max(self.max_sum, price_newpath)

            # 返回节点的最大贡献值
            return node.val + max(left_gain, right_gain)

        max_gain(root)
        return self.max_sum


'''
102. 二叉树的层序遍历

给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
'''
from collections import deque


class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(current_level)

        return result


'''
    3
   / \
  9  20
    /  \
   15   7

首先,我们检查根节点是否为空。在这个例子中,根节点不为空,所以我们继续。
初始化 result = [] 和 queue = deque([root])。此时 queue 中只有根节点 3。
进入 while 循环,因为 queue 不为空。
4. 第一层:
level_size = 1 (queue 中只有一个节点)
创建 current_level = []
从 queue 中取出节点 3,将其值添加到 current_level
将节点 3 的左右子节点(9 和 20)添加到 queue
此时 current_level = [3], queue = deque([9, 20])
将 current_level 添加到 result,现在 result = [[3]]
第二层:
level_size = 2 (queue 中有两个节点)
创建新的 current_level = []
从 queue 中取出节点 9,将其值添加到 current_level
节点 9 没有子节点,不需要添加到 queue
从 queue 中取出节点 20,将其值添加到 current_level
将节点 20 的左右子节点(15 和 7)添加到 queue
此时 current_level = [9, 20], queue = deque([15, 7])
将 current_level 添加到 result,现在 result = [[3], [9, 20]]
第三层:
level_size = 2 (queue 中有两个节点)
创建新的 current_level = []
从 queue 中取出节点 15,将其值添加到 current_level
从 queue 中取出节点 7,将其值添加到 current_level
15 和 7 都没有子节点,不需要添加到 queue
此时 current_level = [15, 7], queue 为空
将 current_level 添加到 result,现在 result = [[3], [9, 20], [15, 7]]
queue 为空,while 循环结束。
返回 result。
最终输出: [[3], [9, 20], [15, 7]]
这就是层序遍历的完整过程。代码使用队列(queue)来确保按层从左到右遍历所有节点,并将每一层的节点值存储在单独的列表中。

'''

'''
 236. 二叉树的最近公共祖先:
 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的度尽可能大（一个节点也可以是它自己的祖先）。”
 '''


class Solution:
    def LowestCommonAncestor(self, root, p, q):
        if not root or root == p or root == q:
            return root
        left = self.LowestCommonAncestor(root.left, p, q)
        right = self.LowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        elif left:
            return left
        elif right:
            return right
        else:
            return None


class Solution:
    def maxDepth(self, root):
        if not root:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        return max(left_depth, right_depth) + 1


class Solution:
    def invertTree(self, root):
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root


'''
105. 从前序与中序遍历序列构造二叉树
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点
'''


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])

        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])

        return root


"""
给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
"""


class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        if not nums:
            return 0

        j = 0  # j 指向下一个要放置元素的位置

        for i in range(len(nums)):
            # 如果 j < 2 或者 nums[i] != nums[j-2]，我们可以放置 nums[i]
            if j < 2 or nums[i] != nums[j - 2]:
                nums[j] = nums[i]
                j += 1

        return j


'''
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
'''


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        def reverse(start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        n = len(nums)
        k = k % n  # 处理 k 大于数组长度的情况

        reverse(0, n - 1)  # 翻转整个数组
        reverse(0, k - 1)  # 翻转前 k 个元素
        reverse(k, n - 1)  # 翻转剩余的元素


"""

给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。
"""


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        total_profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                total_profit += prices[i] - prices[i - 1]

        return total_profit


"""
45. 跳跃游戏 II

给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。

每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，可以跳转到任意 nums[i + j] 处:

0 <= j <= nums[i] 
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
"""


class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0

        jumps = 0
        current_max_reach = 0
        next_max_reach = 0

        for i in range(n - 1):
            next_max_reach = max(next_max_reach, i + nums[i])

            if i == current_max_reach:
                jumps += 1
                current_max_reach = next_max_reach

                if current_max_reach >= n - 1:
                    break

        return jumps


'''
68. 文本左右对齐

给定一个单词数组 words 和一个长度 maxWidth ，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。

你应该使用 “贪心算法” 来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。

要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

文本的最后一行为左对齐，且单词之间不插入额外的空格。

注意:

单词是指由非空格字符组成的字符序列。
每个单词的长度大于 0，小于等于 maxWidth。
输入单词数组 words 至少包含一个单词
'''


class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # 存储最终结果的列表
        result = []
        # 当前正在处理的行中的单词列表
        current_line = []
        # 当前行中所有单词的总长度（不包括空格）
        current_length = 0

        def justify_line(line, length, is_last=False):
            # 如果只有一个单词或是最后一行，左对齐处理
            if len(line) == 1 or is_last:
                return ' '.join(line).ljust(maxWidth)

            # 计算需要添加的空格总数
            spaces = maxWidth - length
            # 计算单词之间的间隔数
            gaps = len(line) - 1
            # 计算每个间隔的基本空格数和多余的空格数
            space_between, extra_spaces = divmod(spaces, gaps)

            justified = []
            for i, word in enumerate(line):
                # 添加单词
                justified.append(word)
                # 如果不是最后一个单词，添加空格
                if i < gaps:
                    # 计算需要添加的空格数
                    spaces_to_add = space_between + (1 if i < extra_spaces else 0)
                    justified.append(' ' * spaces_to_add)

            # 将所有元素连接成一个字符串
            return ''.join(justified)

        for word in words:
            # 检查是否可以将单词添加到当前行
            if current_length + len(word) + len(current_line) <= maxWidth:
                current_line.append(word)
                current_length += len(word)
            else:
                # 当前行无法再添加单词，进行对齐处理
                result.append(justify_line(current_line, current_length))
                # 开始新的一行
                current_line = [word]
                current_length = len(word)

        # 处理最后一行
        if current_line:
            result.append(justify_line(current_line, current_length, is_last=True))

        return result


'''
167. 两数之和 II - 输入有序数组

给你一个下标从 1 开始的整数数组 numbers ，该数组已按 非递减顺序排列  ，请你从数组中找出满足相加之和等于目标数 target 的两个数。如果设这两个数分别是 numbers[index1] 和 numbers[index2] ，则 1 <= index1 < index2 <= numbers.length 。

以长度为 2 的整数数组 [index1, index2] 的形式返回这两个整数的下标 index1 和 index2。

你可以假设每个输入 只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。

你所设计的解决方案必须只使用常量级的额外空间。
'''


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            current_sum = numbers[left] + numbers[right]

            if current_sum == target:
                return [left + 1, right + 1]  # 返回的索引要加1，因为题目要求索引从1开始
            elif current_sum < target:
                left += 1  # 如果和小��目标值，左指针右��
            else:
                right -= 1  # 如果和大于目标值，右指针左移

        # 如果没有找到符合条件的两个数，返回空列表（虽然题目保证有解，但为了代码完整性，我们也处理这种情况）
        return []


"""
三数之和
"""


def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum == 0:
                result.append([nums[i], nums[left], nums[right]])

                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif sum < 0:
                left += 1
            else:
                right -= 1
    return result


"""
209. 长度最小的子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其总和大于等于 target 的长度最小的 
子数组
 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 
"""


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # 初始化左指针为0
        left = 0
        # 初始化当前窗口的和为0
        current_sum = 0
        # 初始化最小长度为无穷大
        min_length = float('inf')

        # 使用右指针遍历整个数组
        for right in range(len(nums)):
            # 将当前元素加入窗口和
            current_sum += nums[right]

            # 当窗口和大于等于目标值时,尝试缩小窗口
            while current_sum >= target:
                # 更新最小长度
                min_length = min(min_length, right - left + 1)
                # 从窗口和中减去左指针指向的元素
                current_sum -= nums[left]
                # 左指针右移
                left += 1

        # 如果找到了符合条件的子数组,返回最小长度;否则返回0
        return min_length if min_length != float('inf') else 0


"""
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 
子串
 的长度。
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        max_len = 0
        left = 0
        seen = set()
        for right in range(n):
            while s[right] in seen:
                seen.remove(s[left])
                left += 1
            seen.add(s[right])
            max_len = max(max_len, right - left + 1)
        return max_len


"""
30. 串联所有单词的子串

给定一个字符串 s 和一个字符串数组 words。 words 中所有字符串 长度相同。

 s 中的 串联子串 是指一个包含  words 中所有字符串以任意顺序排列连接起来的子串。

例如，如果 words = ["ab","cd","ef"]， 那么 "abcdef"， "abefcd"，"cdabef"， "cdefab"，"efabcd"， 和 "efcdab" 都是串联子串。 "acdbef" 不是串联子串，因为他不是任何 words 排列的连接。
返回所有串联子串在 s 中的开始索引。你可以以 任意顺序 返回答案。
"""


class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        # 处理边界情况
        if not s or not words:
            return []  # 如果输入的字符串或单词列表为空，直接返回空列表

        # 初始化关键变量
        result = []  # 存储所有匹配的起始索引

        # 计算关键长度
        word_length = len(words[0])  # 单个单词的长度
        word_count = len(words)  # 单词的总数
        total_length = word_length * word_count  # 所有单词组合在一起的总长度
        str_length = len(s)  # 原始字符串的长度

        # 创建单词频率字典
        word_freq = {}
        for word in words:
            # 统计每个单词出现的次数，例如 ["foo","foo"] 会得到 {"foo": 2}
            word_freq[word] = word_freq.get(word, 0) + 1

        # 遍历所有可能的起始位置
        for i in range(str_length - total_length + 1):
            # 为每个起始位置创建一个新的频率字典副本
            current_freq = word_freq.copy()

            # 标记是否找到匹配
            matched = True

            # 检查从当前位置开始的每个单词
            for j in range(word_count):
                # 计算当前单词的起始位置
                start_pos = i + j * word_length
                # 提取当前单词
                current_word = s[start_pos:start_pos + word_length]

                # 检查当前单词是否有效
                if current_word not in current_freq or current_freq[current_word] == 0:
                    # 如果单词不在字典中或者已经用完了，标记为不匹配
                    matched = False
                    break

                # 减少当前单词的可用次数
                current_freq[current_word] -= 1

            # 如果所有单词都匹配成功
            if matched:
                # 将起始索引添加到结果列表
                result.append(i)

        return result


"""
48. 旋转图像

给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
"""


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)

        # 步骤1: 沿主对角线翻转
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        # 步骤2: 左右翻转
        for i in range(n):
            matrix[i].reverse()


"""
73. 矩阵置零

给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
"""


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m = len(matrix)  # 行数
        n = len(matrix[0])  # 列数

        # 使用第一行和第一列作为标记数组
        first_row_has_zero = False
        first_col_has_zero = False

        # 检查第一行是否有0
        for j in range(n):
            if matrix[0][j] == 0:
                first_row_has_zero = True
                break

        # 检查第一列是否有0
        for i in range(m):
            if matrix[i][0] == 0:
                first_col_has_zero = True
                break

        # 使用第一行和第一列来标记其他行列是否需要置零
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0  # 标记该行需要置零
                    matrix[0][j] = 0  # 标记该列需要置零

        # 根据第一行和第一列的标记来置零
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        # 如果第一行原本有0，则将第一行置零
        if first_row_has_zero:
            for j in range(n):
                matrix[0][j] = 0

        # 如果第一列原本有0，则将第一列置零
        if first_col_has_zero:
            for i in range(m):
                matrix[i][0] = 0


"""
289. 生命游戏

根据 百度百科 ， 生命游戏 ，简称为 生命 ，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。

给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态： 1 即为 活细胞 （live），或 0 即为 死细胞 （dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：

如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是 同时 发生的。给你 m x n 网格面板 board 的当前状态，返回下一个状态。

给定当前 board 的状态，更新 board 到下一个状态。

注意 你不需要返回任何东西。
"""


class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows = len(board)
        cols = len(board[0])

        # 定义八个方向的偏移量
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # 第一次遍历：标记状态变化
        for i in range(rows):
            for j in range(cols):
                # 统计周围活细胞的数量
                live_neighbors = 0

                # 检查八个相邻位置
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        # 只统计原始状态为1的细胞（活细胞）
                        if board[ni][nj] == 1 or board[ni][nj] == 2:
                            live_neighbors += 1

                # 应用生命游戏规则
                if board[i][j] == 1:  # 当前是活细胞
                    if live_neighbors < 2 or live_neighbors > 3:
                        board[i][j] = 2  # 活细胞死亡
                elif board[i][j] == 0:  # 当前是死细胞
                    if live_neighbors == 3:
                        board[i][j] = 3  # 死细胞复活

        # 第二次遍历：更新最终状态
        for i in range(rows):
            for j in range(cols):
                board[i][j] = board[i][j] % 2


"""
228. 汇总区间

给定一个  无重复元素 的 有序 整数数组 nums 。

返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表 。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。

列表中的每个区间范围 [a,b] 应该按如下格式输出：

"a->b" ，如果 a != b
"a" ，如果 a == b

"""


def summaryRanges(self, nums: List[int]) -> List[str]:
    # nums = [0,1,2,4,5,7]
    if not nums:
        return []

    result = []
    start = nums[0]  # start = 0

    # 让我们逐步遍历数组
    for i in range(len(nums)):
        # 第一次遍历：i = 0
        # nums[0] = 0, nums[1] = 1
        # 0+1 == 1，说明连续，继续遍历

        # 第二次遍历：i = 1
        # nums[1] = 1, nums[2] = 2
        # 1+1 == 2，说明连续，继续遍历

        # 第三次遍历：i = 2
        # nums[2] = 2, nums[3] = 4
        # 2+1 != 4，不连续！需要结束当前区间
        # start = 0, nums[2] = 2
        # 添加 "0->2" 到 result
        # 设置新的 start = 4

        # 第四次遍历：i = 3
        # nums[3] = 4, nums[4] = 5
        # 4+1 == 5，说明连续，继续遍历

        # 第五次遍历：i = 4
        # nums[4] = 5, nums[5] = 7
        # 5+1 != 7，不连续！需要结束当前区间
        # start = 4, nums[4] = 5
        # 添加 "4->5" 到 result
        # 设置新的 start = 7

        # 第六次遍历：i = 5
        # 到达数组末尾，需要结束当前区间
        # start = 7, nums[5] = 7
        # 添加 "7" 到 result

        if i == len(nums) - 1 or nums[i] + 1 != nums[i + 1]:
            if start == nums[i]:
                result.append(str(start))
            else:
                result.append(f"{start}->{nums[i]}")

            if i < len(nums) - 1:
                start = nums[i + 1]

    return result


"""
57. 插入区间

给你一个 无重叠的 ，按照区间起始端点排序的区间列表 intervals，其中 intervals[i] = [starti, endi] 表示第 i 个区间的开始和结束，并且 intervals 按照 starti 升序排列。同样给定一个区间 newInterval = [start, end] 表示另一个区间的开始和结束。

在 intervals 中插入区间 newInterval，使得 intervals 依然按照 starti 升序排列，且区间之间不重叠（如果有必要的话，可以合并区间）。

返回插入之后的 intervals。

注意 你不需要原地修改 intervals。你可以创建一个新数组然后返回它。
"""


def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    result = []
    i = 0
    n = len(intervals)

    # 1. 添加所有在新区间之前的不重叠区间
    # 判断条件：当前区间的结束点 < 新区间的起始点
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # 2. 合并所有重叠的区间
    # 判断条件：当前区间的起始点 <= 新区间的结束点
    # 由于上一步已经处理了所有在新区间之前的区间
    # 所以这里只需要判断一个条件就够了
    while i < n and intervals[i][0] <= newInterval[1]:
        # 更新新区间的范围
        # 起始点取两者的最小值
        newInterval[0] = min(newInterval[0], intervals[i][0])
        # 结束点取两者的最大值
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1

    # 3. 添加合并后的新区间
    result.append(newInterval)

    # 4. 添加所有剩余的区间（这些区间都在新区间之后）
    while i < n:
        result.append(intervals[i])
        i += 1

    return result


"""

452. 用最少数量的箭引爆气球

有一些球形气球贴在一堵用 XY 平面表示的墙面上。墙面上的气球记录在整数数组 points ，其中points[i] = [xstart, xend] 表示水平直径在 xstart 和 xend之间的气球。你不知道气球的确切 y 坐标。

一支弓箭可以沿着 x 轴从不同点 完全垂直 地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被 引爆 。可以射出的弓箭的数量 没有限制 。 弓箭一旦被射出之后，可以无限地前进。

给你一个数组 points ，返回引爆所有气球所必须射出的 最小 弓箭数 。
"""


def findMinArrowShots(points: List[List[int]]) -> int:
    if not points:
        return 0

    # 按结束位置排序
    points.sort(key=lambda x: x[1])

    count = 1  # 至少需要一支箭
    pos = points[0][1]  # 第一支箭的位置（第一个气球的结束位置）

    # 遍历剩余的气球
    for start, end in points:
        # 如果当前气球的开始位置大于上一支箭的位置
        # 说明需要一支新箭
        if start > pos:
            count += 1
            pos = end

    return count


"""

71. 简化路径

给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为 更加简洁的规范路径。

在 Unix 风格的文件系统中规则如下：

一个点 '.' 表示当前目录本身。
此外，两个点 '..' 表示将目录切换到上一级（指向父目录）。
任意多个连续的斜杠（即，'//' 或 '///'）都被视为单个斜杠 '/'。
任何其他格式的点（例如，'...' 或 '....'）均被视为有效的文件/目录名称。
返回的 简化路径 必须遵循下述格式：

始终以斜杠 '/' 开头。
两个目录名之间必须只有一个斜杠 '/' 。
最后一个目录名（如果存在）不能 以 '/' 结尾。
此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
返回简化后得到的 规范路径 。
"""


def simplifyPath(path: str) -> str:
    # 使用栈来存储路径中的有效目录名
    stack = []

    # 将路径按'/'分割成目录名列表
    # 例如: "/home//foo/" -> ['', 'home', '', 'foo', '']
    components = path.split('/')

    # 遍历每个目录名
    for component in components:
        # 跳过空字符串和当前目录符号'.'
        if component == '' or component == '.':
            continue

        # 如果遇到'..'且栈不为空,弹出栈顶元素(返回上一级目录)
        elif component == '..':
            if stack:
                stack.pop()

        # 其他情况都是有效的目录名,直接入栈
        else:
            stack.append(component)

    # 将栈中的目录名用'/'连接,并在开头加上'/'
    return '/' + '/'.join(stack)


"""

155. 最小栈

设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。
"""


class MinStack:
    def __init__(self):
        # 初始化两个栈
        self.stack = []  # 主栈,存储所有元素
        self.min_stack = []  # 辅助栈,存储最小值

    def push(self, val: int) -> None:
        # 将元素压入主栈
        self.stack.append(val)

        # 如果辅助栈为空或当前值小于等于辅助栈顶元素
        # 则将当前值也压入辅助栈
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        # 如果主栈顶元素等于辅助栈顶元素
        # 说明要移除的是当前最小值,辅助栈也要弹出
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self) -> int:
        # 返回主栈顶元素
        return self.stack[-1]

    def getMin(self) -> int:
        # 返回辅助栈顶元素(当前最小值)
        return self.min_stack[-1]


'''
150. 逆波兰表达式求值

给你一个字符串数组 tokens ，表示一个根据 逆波兰表示法 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

注意：

有效的算符为 '+'、'-'、'*' 和 '/' 。
每个操作数（运算对象）都可以是一个整数或者另一个表达式。
两个整数之间的除法总是 向零截断 。
表达式中不含除零运算。
输入是一个根据逆波兰表示法表示的算术表达式。
答案及所有中间计算结果可以用 32 位 整数表示。
 

示例 1：

输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
示例 2：

输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
示例 3：

输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
 

提示：

1 <= tokens.length <= 104
tokens[i] 是一个算符（"+"、"-"、"*" 或 "/"），或是在范围 [-200, 200] 内的一个整数
 

逆波兰表达式：

逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。

平常使用的算式则是一种中缀表达式，如 ( 1 + 2 ) * ( 3 + 4 ) 。
该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 。
逆波兰表达式主要有以下两个优点：

去掉括号后表达式无歧义，上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。
适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中
'''


def evalRPN(tokens: List[str]) -> int:
    stack = []
    # 定义运算符对应的操作
    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: int(x / y)  # 注意这里要求向零截断
    }

    for token in tokens:
        if token in operations:
            # 遇到运算符时,从栈中弹出两个数进行计算
            num2 = stack.pop()  # 注意: 后弹出的是第二个操作数
            num1 = stack.pop()  # 先弹出的是第一个操作数
            # 计算结果并压入栈中
            result = operations[token](num1, num2)
            stack.append(result)
        else:
            # 遇到数字时,将其转换为整数后压入栈中
            stack.append(int(token))

    # 最后栈中只剩下一个数,即为最终结果
    return stack[0]


'''
224. 基本计算器

给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。

注意:不允许使用任何将字符串作为数学表达式计算的内置函数，比如 eval() 。

 

示例 1：

输入：s = "1 + 1"
输出：2
示例 2：

输入：s = " 2-1 + 2 "
输出：3
示例 3：

输入：s = "(1+(4+5+2)-3)+(6+8)"
输出：23
'''


def calculate(s: str) -> int:
    stack = []  # 用于存储计算过程中的数字和符号
    num = 0  # 当前正在处理的数字
    sign = 1  # 当前数字的符号(1表示正,-1表示负)
    result = 0  # 最终结果

    for char in s:
        # 如果是数字,则累加到num中
        if char.isdigit():
            num = num * 10 + int(char)

        # 如果是加号,则将之前的数字*符号加到结果中,并重置符号为正
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1

        # 如果是减号,则将之前的数字*符号加到结果中,并重置符号为负
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1

        # 如果是左括号,将当前结果和符号压入栈中
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            # 重置结果和符号
            result = 0
            sign = 1

        # 如果是右括号,计算括号内的结果
        elif char == ')':
            result += sign * num
            num = 0
            # 将结果与栈顶的符号相乘
            result *= stack.pop()
            # 加上栈顶的数字
            result += stack.pop()

    # 处理最后一个数字
    result += sign * num

    return result


'''
100. 相同的树

给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
'''


class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        # 如果两个节点都为空,返回True
        if not p and not q:
            return True

        # 如果其中一个节点为空而另一个不为空,返回False 
        if not p or not q:
            return False

        # 如果两个节点的值不相等,返回False
        if p.val != q.val:
            return False

        # 递归比较左子树和右子树
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


'''
101. 对称二叉树
简单
相关标签
相关企业
给你一个二叉树的根节点 root ， 检查它是否轴对称

'''


class Solution:
    def isSymmetric(self, root):
        if not root:
            return True

        queue = deque()
        # 直接将左右子节点放入队列
        queue.append(root.left)
        queue.append(root.right)

        while queue:
            # 每次取出两个节点进行比较
            left = queue.popleft()
            right = queue.popleft()

            # 如果两个节点都为空，继续检查下一对节点
            if not left and not right:
                continue
            # 如果其中一个节点为空，或者节点值不相等，则不对称
            if not left or not right or left.val != right.val:
                return False

            # 将对应的子节点加入队列
            queue.append(left.left)
            queue.append(right.right)
            queue.append(left.right)
            queue.append(right.left)

        return True


#   递归法
def isSymmetric(root):
    # 如果树为空，则认为是对称的
    if not root:
        return True

    def isMirror(left, right):
        # 如果两个节点都为空，则对称
        if not left and not right:
            return True
        # 如果其中一个节点为空，则不对称
        if not left or not right:
            return False

        # 判断条件：
        # 1. 当前节点的值相等
        # 2. 左子树的左节点和右子树的右节点对称
        # 3. 左子树的右节点和右子树的左节点对称
        return (left.val == right.val and
                isMirror(left.left, right.right) and
                isMirror(left.right, right.left))

    # 从根节点的左右子树开始判断
    return isMirror(root.left, root.right)


'''
106. 从中序与后序遍历序列构造二叉树
中等
相关标签
相关企业
给定两个整数数组 inorder 和 postorder ，其中 inorder 是二叉树的中序遍历， postorder 是同一棵树的后序遍历，请你构造并返回这颗 二叉树 。

'''


class TreeNode:
    def __init__(self, val=0, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, inorder: list[int], postorder: list[int]) -> TreeNode:
        if not inorder or not postorder:
            return None
        root_val = postorder[-1]
        root = TreeNode(root_val)
        mid = inorder.index(root_val)
        root.left = self.buildTree(inorder[:mid], postorder[:mid])
        root.right = self.buildTree(inorder[mid + 1:], postorder[mid:-1])
        return root


'''
117. 填充每个节点的下一个右侧节点指针 II

给定一个二叉树：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL 。

初始状态下，所有 next 指针都被设置为 NULL 。
'''


# 迭代法
class Solution:
    def connect(self, root):
        if not root:
            return root
        curr = root
        while curr:
            dummy = Node(0)
            tail = dummy
            while curr:
                if curr.left:
                    tail.next = curr.left
                    tail = tail.next
                if curr.right:
                    tail.next = curr.right
                    tail = tail.next
                curr = curr.next
            curr = dummy.next
        return root


# bfs
from collections import deque


class Solution:
    def connect(self, root):
        if not root:
            return root
        queue = deque([root])
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if i < level_size - 1:
                    node.next = queue[0]

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root


'''
114. 二叉树展开为链表
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

'''


# 前序遍历＋数组存储
class Solution:
    def flatten(self, root: TreeNode) -> None:
        if not root:
            return
        nodes = []

        def preorder(node):
            if not node:
                return
            nodes.append(node)
            preorder(node.left)
            preorder(node.right)

        preorder(root)
        for i in range(1, len(nodes)):
            nodes[i - 1].left = None
            nodes[i - 1].right = nodes[i]


# 寻找前驱节点法：
class Solution:
    def flatten(self, root: TreeNode) -> None:
        if not root:
            return
        curr = root
        while curr:
            if curr.left:
                predecessor = curr.left
                while predecessor.right:
                    predecessor = predecessor.right
                predecessor.right = curr.right
                curr.right = curr.left
                curr.left = None
            curr = curr.right


'''
112. 路径总和

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。

叶子节点 是指没有子节点的节点。
 
 '''


# dfs

class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return root.val == targetSum

        return (self.hasPathSum(root.left, targetSum - root.val) or
                self.hasPathSum(root.right, targetSum - root.val)
                )


# bfs
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if not root:
            return False
        queue = deque([(root, root.val)])
        while queue:
            node, curr_sum = queue.popleft()
            if not node.left and not node.right and curr_sum == targetSum:
                return True
            if node.left:
                queue.append((node.left, curr_sum + node.left.val))
            if node.right:
                queue.append((node.right, curr_sum + node.right.val))
        return False


'''
124 .   二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。
'''


class Solution:
    def maxPathSum(self, root):
        self.max_sum = float('-inf')

        def max_gain(node):
            if not node:
                return 0

            # 递归计算左右子树的最大贡献值
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # 计算当前节点的最大路径和
            price_newpath = node.val + left_gain + right_gain

            # 更新全局最大路径和
            self.max_sum = max(self.max_sum, price_newpath)

            # 返回节点的最大贡献值
            return node.val + max(left_gain, right_gain)

        max_gain(root)
        return self.max_sum


'''
102. 二叉树的层序遍历

给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
'''
from collections import deque


class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(current_level)

        return result


'''
    3
   / \
  9  20
    /  \
   15   7

首先,我们检查根节点是否为空。在这个例子中,根节点不为空,所以我们继续。
初始化 result = [] 和 queue = deque([root])。此时 queue 中只有根节点 3。
进入 while 循环,因为 queue 不为空。
4. 第一层:
level_size = 1 (queue 中只有一个节点)
创建 current_level = []
从 queue 中取出节点 3,将其值添加到 current_level
将节点 3 的左右子节点(9 和 20)添加到 queue
此时 current_level = [3], queue = deque([9, 20])
将 current_level 添加到 result,现在 result = [[3]]
第二层:
level_size = 2 (queue 中有两个节点)
创建新的 current_level = []
从 queue 中取出节点 9,将其值添加到 current_level
节点 9 没有子节点,不需要添加到 queue
从 queue 中取出节点 20,将其值添加到 current_level
将节点 20 的左右子节点(15 和 7)添加到 queue
此时 current_level = [9, 20], queue = deque([15, 7])
将 current_level 添加到 result,现在 result = [[3], [9, 20]]
第三层:
level_size = 2 (queue 中有两个节点)
创建新的 current_level = []
从 queue 中取出节点 15,将其值添加到 current_level
从 queue 中取出节点 7,将其值添加到 current_level
15 和 7 都没有子节点,不需要添加到 queue
此时 current_level = [15, 7], queue 为空
将 current_level 添加到 result,现在 result = [[3], [9, 20], [15, 7]]
queue 为空,while 循环结束。
返回 result。
最终输出: [[3], [9, 20], [15, 7]]
这就是层序遍历的完整过程。代码使用队列(queue)来确保按层从左到右遍历所有节点,并将每一层的节点值存储在单独的列表中。

'''

'''
 236. 二叉树的最近公共祖先:
 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的度尽可能大（一个节点也可以是它自己的祖先）。”
 '''


class Solution:
    def LowestCommonAncestor(self, root, p, q):
        if not root or root == p or root == q:
            return root
        left = self.LowestCommonAncestor(root.left, p, q)
        right = self.LowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        elif left:
            return left
        elif right:
            return right
        else:
            return None


class Solution:
    def maxDepth(self, root):
        if not root:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        return max(left_depth, right_depth) + 1


class Solution:
    def invertTree(self, root):
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root


'''
105. 从前序与中序遍历序列构造二叉树
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点
'''


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])

        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])

        return root


"""
给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
"""


class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        if not nums:
            return 0

        j = 0  # j 指向下一个要放置元素的位置

        for i in range(len(nums)):
            # 如果 j < 2 或者 nums[i] != nums[j-2]，我们可以放置 nums[i]
            if j < 2 or nums[i] != nums[j - 2]:
                nums[j] = nums[i]
                j += 1

        return j


'''
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
'''


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        def reverse(start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        n = len(nums)
        k = k % n  # 处理 k 大于数组长度的情况

        reverse(0, n - 1)  # 翻转整个数组
        reverse(0, k - 1)  # 翻转前 k 个元素
        reverse(k, n - 1)  # 翻转剩余的元素


"""

给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。
"""


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        total_profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                total_profit += prices[i] - prices[i - 1]

        return total_profit


"""
45. 跳跃游戏 II

给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。

每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，可以跳转到任意 nums[i + j] 处:

0 <= j <= nums[i] 
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
"""


class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0

        jumps = 0
        current_max_reach = 0
        next_max_reach = 0

        for i in range(n - 1):
            next_max_reach = max(next_max_reach, i + nums[i])

            if i == current_max_reach:
                jumps += 1
                current_max_reach = next_max_reach

                if current_max_reach >= n - 1:
                    break

        return jumps


'''
68. 文本左右对齐

给定一个单词数组 words 和一个长度 maxWidth ，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。

你应该使用 “贪心算法” 来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。

要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

文本的最后一行为左对齐，且单词之间不插入额外的空格。

注意:

单词是指由非空格字符组成的字符序列。
每个单词的长度大于 0，小于等于 maxWidth。
输入单词数组 words 至少包含一个单词
'''


class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # 存储最终结果的列表
        result = []
        # 当前正在处理的行中的单词列表
        current_line = []
        # 当前行中所有单词的总长度（不包括空格）
        current_length = 0

        def justify_line(line, length, is_last=False):
            # 如果只有一个单词或是最后一行，左对齐处理
            if len(line) == 1 or is_last:
                return ' '.join(line).ljust(maxWidth)

            # 计算需要添加的空格总数
            spaces = maxWidth - length
            # 计算单词之间的间隔数
            gaps = len(line) - 1
            # 计算每个间隔的基本空格数和多余的空格数
            space_between, extra_spaces = divmod(spaces, gaps)

            justified = []
            for i, word in enumerate(line):
                # 添加单词
                justified.append(word)
                # 如果不是最后一个单词，添加空格
                if i < gaps:
                    # 计算需要添加的空格数
                    spaces_to_add = space_between + (1 if i < extra_spaces else 0)
                    justified.append(' ' * spaces_to_add)

            # 将所有元素连接成一个字符串
            return ''.join(justified)

        for word in words:
            # 检查是否可以将单词添加到当前行
            if current_length + len(word) + len(current_line) <= maxWidth:
                current_line.append(word)
                current_length += len(word)
            else:
                # 当前行无法再添加单词，进行对齐处理
                result.append(justify_line(current_line, current_length))
                # 开始新的一行
                current_line = [word]
                current_length = len(word)

        # 处理最后一行
        if current_line:
            result.append(justify_line(current_line, current_length, is_last=True))

        return result


'''
167. 两数之和 II - 输入有序数组

给你一个下标从 1 开始的整数数组 numbers ，该数组已按 非递减顺序排列  ，请你从数组中找出满足相加之和等于目标数 target 的两个数。如果设这两个数分别是 numbers[index1] 和 numbers[index2] ，则 1 <= index1 < index2 <= numbers.length 。

以长度为 2 的整数数组 [index1, index2] 的形式返回这两个整数的下标 index1 和 index2。

你可以假设每个输入 只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。

你所设计的解决方案必须只使用常量级的额外空间。
'''


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            current_sum = numbers[left] + numbers[right]

            if current_sum == target:
                return [left + 1, right + 1]  # 返回的索引要加1，因为题目要求索引从1开始
            elif current_sum < target:
                left += 1  # 如果和小��目标值，左指针右��
            else:
                right -= 1  # 如果和大于目标值，右指针左移

        # 如果没有找到符合条件的两个数，返回空列表（虽然题目保证有解，但为了代码完整性，我们也处理这种情况）
        return []


"""
三数之和
"""


def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum == 0:
                result.append([nums[i], nums[left], nums[right]])

                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif sum < 0:
                left += 1
            else:
                right -= 1
    return result


"""
209. 长度最小的子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其总和大于等于 target 的长度最小的 
子数组
 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 
"""


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # 初始化左指针为0
        left = 0
        # 初始化当前窗口的和为0
        current_sum = 0
        # 初始化最小长度为无穷大
        min_length = float('inf')

        # 使用右指针遍历整个数组
        for right in range(len(nums)):
            # 将当前元素加入窗口和
            current_sum += nums[right]

            # 当窗口和大于等于目标值时,尝试缩小窗口
            while current_sum >= target:
                # 更新最小长度
                min_length = min(min_length, right - left + 1)
                # 从窗口和中减去左指针指向的元素
                current_sum -= nums[left]
                # 左指针右移
                left += 1

        # 如果找到了符合条件的子数组,返回最小长度;否则返回0
        return min_length if min_length != float('inf') else 0


"""
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 
子串
 的长度。
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        max_len = 0
        left = 0
        seen = set()
        for right in range(n):
            while s[right] in seen:
                seen.remove(s[left])
                left += 1
            seen.add(s[right])
            max_len = max(max_len, right - left + 1)
        return max_len


"""
30. 串联所有单词的子串

给定一个字符串 s 和一个字符串数组 words。 words 中所有字符串 长度相同。

 s 中的 串联子串 是指一个包含  words 中所有字符串以任意顺序排列连接起来的子串。

例如，如果 words = ["ab","cd","ef"]， 那么 "abcdef"， "abefcd"，"cdabef"， "cdefab"，"efabcd"， 和 "efcdab" 都是串联子串。 "acdbef" 不是串联子串，因为他不是任何 words 排列的连接。
返回所有串联子串在 s 中的开始索引。你可以以 任意顺序 返回答案。
"""


class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        # 处理边界情况
        if not s or not words:
            return []  # 如果输入的字符串或单词列表为空，直接返回空列表

        # 初始化关键变量
        result = []  # 存储所有匹配的起始索引

        # 计算关键长度
        word_length = len(words[0])  # 单个单词的长度
        word_count = len(words)  # 单词的总数
        total_length = word_length * word_count  # 所有单词组合在一起的总长度
        str_length = len(s)  # 原始字符串的长度

        # 创建单词频率字典
        word_freq = {}
        for word in words:
            # 统计每个单词出现的次数，例如 ["foo","foo"] 会得到 {"foo": 2}
            word_freq[word] = word_freq.get(word, 0) + 1

        # 遍历所有可能的起始位置
        for i in range(str_length - total_length + 1):
            # 为每个起始位置创建一个新的频率字典副本
            current_freq = word_freq.copy()

            # 标记是否找到匹配
            matched = True

            # 检查从当前位置开始的每个单词
            for j in range(word_count):
                # 计算当前单词的起始位置
                start_pos = i + j * word_length
                # 提取当前单词
                current_word = s[start_pos:start_pos + word_length]

                # 检查当前单词是否有效
                if current_word not in current_freq or current_freq[current_word] == 0:
                    # 如果单词不在字典中或者已经用完了，标记为不匹配
                    matched = False
                    break

                # 减少当前单词的可用次数
                current_freq[current_word] -= 1

            # 如果所有单词都匹配成功
            if matched:
                # 将起始索引添加到结果列表
                result.append(i)

        return result


'''
129. 求根节点到叶节点数字之和

给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。

'''


class Solution:
    def sumNumbers(self, root) -> int:
        def dfs(node, current_num):
            if not node:
                return 0

            # 计算当前路径的数字
            current_num = current_num * 10 + node.val

            # 如果是叶子节点，返回当前数字
            if not node.left and not node.right:
                return current_num

            # 递归计算左右子树的和
            return dfs(node.left, current_num) + dfs(node.right, current_num)

        return dfs(root, 0)


'''


'''
