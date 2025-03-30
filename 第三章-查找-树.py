# 二分查找算法
from typing import List


def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# test
# arr = [1, 3, 5, 7, 9]
# target = 7
# print(binary_search(arr, target))  # output: 3


# 二叉查找树实现
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def searchBST(root: TreeNode, val: int) -> TreeNode:
    if not root:
        return None
    if root.val == val:
        return root
    elif root.val < val:
        return searchBST(root.left, val)
    else:
        return searchBST(root.right, val)


# floor和ceil实现
def floor(root: TreeNode, val: int):
    if not root:
        return None
    if root.val == val:
        return root
    elif root.val < val:
        return floor(root.right, val)

    node = floor(root.left, val)
    if node:
        return node
    else:
        return root


def ceil(root: TreeNode, val: int):
    if not root:
        return None
    if root.val == val:
        return root
    elif root.val > val:
        return ceil(root.left, val)
    node = ceil(root.right, val)
    if node:
        return node
    else:
        return root


# select和rank实现
def get_size(left):
    if not left:
        return 0
    return get_size(left.left)


def select(root: TreeNode, k: int):
    if not root:
        return None
    left_size = get_size(root.left)
    if k == left_size + 1:
        return root.val
    elif k < left_size + 1:
        return select(root.left, k)
    else:
        return select(root.right, k - left_size - 1)


def rank(root: TreeNode, val: int):
    if not root:
        return 0
    if root.val == val:
        return get_size(root.left)
    elif root.val < val:
        return rank(root.right, val) + get_size(root.left) + 1
    else:
        return rank(root.left, val)


# delete实现
def deleteNode(root: TreeNode, key: int) -> TreeNode:
    if not root:
        return root
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left:
            temp = root.right
            root = None
            return temp
        elif not root.right:
            temp = root.left
            root = None
            return temp
        temp = minValueNode(root.right)
        root.val = temp.val
        root.right = deleteNode(root.right, temp.val)
    return root


def minValueNode(node: TreeNode) -> TreeNode:
    current = node
    while current.left:
        current = current.left
    return current


# # test
# root = TreeNode(4)
# root.left = TreeNode(2)
# root.right = TreeNode(6)
# root.left.left = TreeNode(1)
# root.left.right = TreeNode(3)
# root.right.left = TreeNode(5)
# root.right.right = TreeNode(7)
#
#
#
#
#
# R = deleteNode(root, 3)
# print(R.val)  # output: 4
# t = deleteNode(root,4)
# print(t.val)  # output: 5
# print(t.left.val)
# print(t.right.val)  # output: 6 7


# 二叉查找树的范围查找
def searchRange(root: TreeNode, low: int, high: int):
    res = []
    if not root:
        return res
    if root.val >= low and root.val <= high:
        res.append(root.val)
    if root.val > low:
        res += searchRange(root.left, low, high)
    if root.val < high:
        res += searchRange(root.right, low, high)
    return res


# test
root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(6)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
root.right.left = TreeNode(5)
root.right.right = TreeNode(7)
print(searchRange(root, 2, 6))  # output: [2, 3, 4, 5, 6]


# 红黑树的插入算法
class Node:
    def __init__(self, val=0, color='red', left=None, right=None):
        self.val = val
        self.color = color
        self.left = left
        self.right = right


def insert(root, val):
    if not root:
        return Node(val)
    if val < root.val:
        root.left = insert(root.left, val)

    else:
        root.right = insert(root.right, val)


def rotateLeft(node):
    new_root = node.right
    node.right = new_root.left
    new_root.left = node

    new_root.color = node.color
    node.color = 'red'
    return new_root


def rotateRight(node):
    new_root = node.left
    node.left = new_root.right
    new_root.right = node
    new_root.color = node.color
    node.color = 'red'
    return new_root


def flipColors(node):
    node.color = 'red'
    node.left.color = 'black'
    node.right.color = 'black'


# 红黑树的插入算法
def insertRBTree(root, val):
    if not root:
        return Node(val)
    if val < root.val:
        if not root.left:
            root.left = Node(val)
        else:
            insertRBTree(root.left, val)
    else:
        if not root.right:
            root.right = Node(val)

        else:
            insertRBTree(root.right, val)

    if root.left and root.left.color == 'red' and root.right and root.right.color == 'red':
        flipColors(root)

    if root.left and root.left.color == 'red' and root.left.left and root.left.left.color == 'red':
        root = rotateRight(root)

    if root.right and root.right.color == 'red' and root.left and root.left.color != 'red':
        root = rotateLeft(root)

    root.len = get_size(root.left) + get_size(root.right) + 1
    return root



'''
基于拉链法的散列表
'''
class ListNode:
    def __init__(self, key=0, val=0, next=None):
        self.key = key
        self.val = val
        self.next = next

class MyHashMap:
    def __init__(self):
        self.size = 1000
        self.table = [None] * self.size



    def put(self, key: int, value: int) -> None:
        index = key % self.size
        if not self.table[index]:
            self.table[index] = ListNode(key, value)
        else:
            curr = self.table[index]
            while curr.next:
                if curr.key == key:
                    curr.val = value
                    return
                curr = curr.next
            curr.next = ListNode(key, value)

    def get(self, key: int) -> int:
        index = key % self.size
        curr = self.table[index]
        while curr:
            if curr.key == key:
                return curr.val
            curr = curr.next
        return -1

    def remove(self, key: int) -> None:
        index = key % self.size
        curr = self.table[index]
        prev = None
        while curr:
            if curr.key == key:
                if not prev:
                    self.table[index] = curr.next
                else:
                    prev.next = curr.next
                return
            prev = curr
            curr = curr.next


