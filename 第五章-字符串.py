'''
键索引计数法：

'''
def key_index_count(s):
    count = [0] * 26
    for c in s:
        if c.isalpha():
            count[ord(c.lower()) - ord('a')] += 1
    return count

s = "hello world"
print(key_index_count(s))


'''
    低位优先的字符串排序
'''


def lsd_radix_sort(strings, W):
    """
    低位优先的字符串排序 (LSD Radix Sort) 适用于固定长度的字符串。

    :param strings: 要排序的字符串列表
    :param W: 字符串的长度（所有字符串必须为此长度）
    :return: 排序后的字符串列表
    """
    N = len(strings)
    R = 256  # ASCII 字符集大小
    aux = [''] * N

    for d in range(W - 1, -1, -1):
        count = [0] * (R + 1)

        # 计算每个字符的频率
        for i in range(N):
            count[ord(strings[i][d]) + 1] += 1

        # 将频率转换为索引
        for r in range(R):
            count[r + 1] += count[r]

        # 按照索引分配字符串
        for i in range(N):
            aux[count[ord(strings[i][d])]] = strings[i]
            count[ord(strings[i][d])] += 1

        # 将排序后的结果复制回原数组
        for i in range(N):
            strings[i] = aux[i]


    return strings


# 示例用法
strings = ["dab", "cab", "fad", "bad", "dad", "ebb", "ace", "add"]
sorted_strings = lsd_radix_sort(strings, 3)
print(sorted_strings)



'''
    高位优先的字符串排序
'''


def char_at(s, d):
    """返回字符串s在第d位置的字符的ASCII值，如果d大于字符串长度，返回-1"""
    if d < len(s):
        return ord(s[d])
    else:
        return -1


def msd_sort(strings, low, high, d):
    """对字符串数组strings进行MSD排序"""
    if high <= low:
        return

    R = 256  # 字符集大小（ASCII）
    count = [0] * (R + 2)  # 计数数组

    # 计算频率
    for i in range(low, high + 1):
        c = char_at(strings[i], d)
        count[c + 2] += 1

    # 将频率转化为索引
    for r in range(R + 1):
        count[r + 1] += count[r]

    # 数据分类
    aux = [None] * (high - low + 1)
    for i in range(low, high + 1):
        c = char_at(strings[i], d)
        aux[count[c + 1]] = strings[i]
        count[c + 1] += 1

    # 回写
    for i in range(low, high + 1):
        strings[i] = aux[i - low]

    # 递归地对每个字符子集排序
    for r in range(R):
        msd_sort(strings, low + count[r], low + count[r + 1] - 1, d + 1)


def msd_string_sort(strings):
    """外部调用函数，对字符串列表进行MSD排序"""
    msd_sort(strings, 0, len(strings) - 1, 0)


'''
    三向字符串的快速排序

'''


def char_at2(s, d):
    """返回字符串s在第d位置的字符的ASCII值，如果d大于字符串长度，返回-1"""
    if d < len(s):
        return ord(s[d])
    else:
        return -1


def three_way_string_quick_sort(strings, low, high, d):
    """三向字符串快速排序"""
    if high <= low:
        return

    lt, gt = low, high
    v = char_at2(strings[low], d)  # 划分字符（当前第d位）
    i = low + 1

    while i <= gt:
        t = char_at2(strings[i], d)
        if t < v:
            strings[lt], strings[i] = strings[i], strings[lt]
            lt += 1
            i += 1
        elif t > v:
            strings[i], strings[gt] = strings[gt], strings[i]
            gt -= 1
        else:
            i += 1

    # 递归排序小于部分
    three_way_string_quick_sort(strings, low, lt - 1, d)

    # 递归排序等于部分的下一个字符
    if v >= 0:  # 仅在v不为-1时继续对等于部分递归排序
        three_way_string_quick_sort(strings, lt, gt, d + 1)

    # 递归排序大于部分
    three_way_string_quick_sort(strings, gt + 1, high, d)


def three_way_string_sort(strings):
    """外部调用函数，对字符串列表进行三向字符串快速排序"""
    three_way_string_quick_sort(strings, 0, len(strings) - 1, 0)


'''
正则表达式的模式匹配（grep）  构造对应的NFA 的转换有向图
'''
class State:
    def __init__(self,is_final=False):
        self.is_final = is_final
        self.transitions = {}

class NFA:
    def __init__(self):
        self.start_state = State()
        self.states = [self.start_state]

def regex_to_rfa(regex):
    nfa = NFA()
    current_state = nfa.start_state
    i=0
    while i< len(regex):
        char = regex[i]

        if char == '(':
            sub_nfa,end_index = parse_subexpression(regex,i+1)
            current_state.transitions['ε'] = [sub_nfa.start_state]
            current_state = sub_nfa.states[-1]
            i = end_index
        elif char == '|':
            alternative_nfa = NFA()
            alternative_state = alternative_nfa.start_state
            nfa.states.append(alternative_state)
            current_state.transitions['ε'] = [alternative_state]
            current_state = alternative_state
        elif char == '*':
            new_state = State()
            nfa.states.append(new_state)
            current_state.transitions['ε'] = [new_state]
            new_state.transitions['ε'] = [current_state]
            current_state = new_state
        else :
            new_state = State()
            nfa.states.append(new_state)
            if char not in current_state.transitions:
                current_state.transitions[char] = []
            current_state.transitions[char].append(new_state)
            current_state = new_state
        i+=1
    current_state.is_final = True
    return nfa

def parse_subexpression(regex,start_index):
    nfa = NFA()
    current_state = nfa.start_state
    i = start_index
    parentheses_count = 1
    while i < len(regex):
        char = regex[i]
        if char == '(':
            parentheses_count += 1
        elif char == ')':
            parentheses_count -=1
            if parentheses_count == 0:
                current_state.is_final = True
                return nfa,i
        #处理子表达式中的字符，类似于主函数
        if char not in '()|*':
            new_state = State()
            nfa.states.append(new_state)
            if char not in current_state.transitions:
                current_state.transitions[char] = []
            current_state.transitions[char].append(new_state)
            current_state = new_state

        i+=1
    raise ValueError('Unmatched parentheses in regex')




#
# [S0] --a--> [S1] --ε--> [S2] --b--> [S3]
#                 |                    |
#                 |        --ε--       |
#                 |       |      |     |
#                 |       V      V     V
#                 |      [S4] --c--> [S5]
#                 |       ^            |
#                 |       |            |
#                 |       ---ε---------|
#                 |                    |
#                 ----------ε----------
#                 |
#                 v
#                [S6] --d--> [S7]


'''
    暴力子字符串查找算法
'''

def brute_force_substring_search(text,pattern):
    n = len(text)
    m = len(pattern)
    for i in range(n-m+1):
        j=0
        while j<m and text[i+j] == pattern[j]:
            j+=1
        if j == m:
            return i
    return -1


'''
    KMP子字符串查找算法
'''

def kmp_substring_search(text,pattern):
    def compute_pmt(pattern):
        pmt = [0] * len(pattern)
        length = 0
        i = 1
        while i< len(pattern) and pattern[i] == pattern[length]:
            length += 1
            pmt[i] = length
            i+= 1
        if length >0:
            length = pmt[length-1]
        elif length < 0:
            pmt[i] = 0
            i += 1
        return  pmt
    pmt = compute_pmt(pattern)
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i+=1
            j+=1
            if j == len(pattern):
                return i-j
        elif j > 0 :
            j = pmt[j-1]
        else:
            i+=1
    return -1

# 示例用法
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
print(kmp_substring_search(text,pattern))