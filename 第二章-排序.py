# 选择排序
def select_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(select_sort(r))


# 冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(bubble_sort(r))


# 插入排序
def insert_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(insert_sort(r))


# 希尔排序
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j > gap - 1 and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
                arr[j] = temp

        gap //= 2
    return arr


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(shell_sort(r))


# 归并排序(原地归并)
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        merge_sort(left_half)
        merge_sort(right_half)
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1

            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(merge_sort(r))


# 自底向上的归并排序(非原地归并)
def merge_sort_bottom_up(arr):
    n = len(arr)
    temp = [0] * n
    sz = 1
    while sz < n:
        for i in range(0, n - sz, sz + sz):
            merge(arr, i, i + sz - 1, min(i + 2 * sz - 1, n - 1), temp)
        sz += sz
    return arr


def merge(arr, l, m, r, temp):
    i = l
    j = m + 1
    k = l
    while i <= m and j <= r:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1
    while i <= m:
        temp[k] = arr[i]
        i += 1
        k += 1
    while j <= r:
        temp[k] = arr[j]
        j += 1
        k += 1
    for i in range(l, r + 1):
        arr[i] = temp[i]


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(merge_sort_bottom_up(r))


# 自顶向下的归并排序（非原地排序）
def merge_sort_top_down(arr):
    n = len(arr)
    temp = [0] * n

    def merge_sort_helper(arr, temp, l, r):
        if l < r:
            mid = (l + r) // 2
            merge_sort_helper(arr, temp, l, mid)
            merge_sort_helper(arr, temp, mid + 1, r)
            merge(arr, l, mid, r, temp)

    merge_sort_helper(arr, temp, 0, n - 1)
    return arr


def merge(arr, l, m, r, temp):
    i = l
    j = m + 1
    k = l
    while i <= m and j <= r:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1
    while i <= m:
        temp[k] = arr[i]
        i += 1
        k += 1
    while j <= r:
        temp[k] = arr[j]
        j += 1
        k += 1
    for i in range(l, r + 1):
        arr[i] = temp[i]


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(merge_sort_top_down(r))


# 快速排序
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = []
        right = []
        for i in range(1, len(arr)):
            if arr[i] < pivot:
                left.append(arr[i])
            else:
                right.append(arr[i])

    return quick_sort(left) + [pivot] + quick_sort(right)


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(quick_sort(r))


# 三向切分的快速排序
def quick_sort_3way(arr):
    if len(arr) <= 1:
        return arr
    else:
        left, middle, right = partition(arr)
    return quick_sort_3way(left) + middle + quick_sort_3way(right)


def partition(arr):
    pivot = arr[len(arr) // 2]
    left = []
    middle = []
    right = []
    for x in arr:
        if x < pivot:
            left.append(x)
        elif x == pivot:
            middle.append(x)
        else:
            right.append(x)
    return left, middle, right


r = [5, 2, 8, 3, 3, 3, 3, 3, 9, 1, 4, 7, 4, 43, 4, 7, 6, 9, 9, 9]
print(quick_sort_3way(r))



# 堆排序
def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 -1 , -1 ,-1 ):
        heapify(arr,n,i)
    for i in range(n-1,0,-1):
        arr[i] ,arr[0] = arr[0], arr[i]
        heapify(arr,i,0)
    return arr

def heapify(arr,n,i):
    largst = i
    left = 2*i + 1
    right = 2*i +2
    if left < n and arr[left] > arr[largst]:
        largst = left
    if right < n and arr[right] > arr[largst]:
        largst = right
    if largst != i:
        arr[i],arr[largst] = arr[largst],arr[i]
        heapify(arr,n,largst)


r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(heap_sort(r))



r = [5, 2, 8, 3, 9, 1, 7, 6, 4]
print(heap_sort(r))