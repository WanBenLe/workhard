import string

import numpy as np


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


class Stack(object):
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)


def test1(a):
    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1

    r1 = []
    Stack2 = Stack()
    b = a.split()
    for s in b:
        if s in '0123456789':
            r1.append(s)
        elif s == '(':
            Stack2.push(s)
        elif s == ')':
            x1 = Stack2.pop()
            while x1 != '(':
                r1.append(x1)
                x1 = Stack2.pop()
        else:
            while (not Stack2.isEmpty()) and (prec[Stack2.peek()] >= prec[s]):
                r1.append(Stack2.pop())
                r1.append(' ')
            Stack2.push(s)
    while not Stack2.isEmpty():
        r1.append(Stack2.pop())
    return r1


def test2(a):
    stack = Stack()
    for s in a:
        if s in '0123456789':
            stack.push(s)
        else:
            num1 = stack.pop()
            num2 = stack.pop()
            stack.push(str(eval(num1 + s + num2)))
    return int(stack.peek())


def test3(a, b):
    all_l = [0] * (b + 1)
    for i in range(b + 1):
        count = i
        for r in [x for x in a if x <= i]:
            if all_l[i - r] + 1 < count:
                count = all_l[i - r] + 1
        all_l[i] = count
    return all_l[-1]


def test4(a, b):
    mid = len(a) // 2
    if len(a) > 1:
        if a[mid] < b:
            a = a[mid:]
            return test4(a, b)
        elif a[mid] > b:
            a = a[:mid]
            return test4(a, b)
        elif a[mid] == b:
            return 'a'
    else:
        return 'b'


def hashStr(a, b):
    num = 0
    for ind, i in enumerate(a):
        num += ord(i) * (ind + 1)
    return num % b


class hash_table():
    def __init__(self, size):
        self.size = size
        self.items = [] * size


def test5(a, b):
    hash = [None] * b
    for i in a:
        modx = i % b
        if hash[modx] == None:
            hash[modx] = i
        else:
            app = 0
            while hash[modx + app] != None:
                app += 1
                if modx + app == b:
                    app -= b
            if app == 0:
                print('error')
            hash[modx + app] = i
    return hash


def test6(a, b):
    hash = [None] * b
    for i in a:
        modx = i % b
        if hash[modx] == None:
            hash[modx] = i
        else:
            indx = (modx + 3) % b
            while hash[indx] != None:
                if indx == modx:
                    print('error')
                indx = (indx + 3) % b
            hash[indx] = i
    return hash


def test7(a, b):
    hash = [None] * b
    x1 = 0
    for i in a:
        modx = i % b
        if hash[modx] == None:
            hash[modx] = i
        else:
            indx = (modx + 1 + x1 ** 2) % b
            while hash[indx] != None:
                if indx == modx:
                    print('error')
                indx = (indx + 1 + x1 ** 2) % b
            hash[indx] = i
            x1 += 1
    return hash


def test8(a, b):
    hash = {i: [] for i in range(11)}
    for i in a:
        modx = i % b
        hash[modx].append(i)
    return hash


def test9(a):
    n = len(a)
    for i in range(n):
        flag = 0
        for j in range(n - i - 1):
            if a[j + i + 1] < a[i]:
                a[j + i + 1], a[i] = a[i], a[j + i + 1]
                flag = 1
        if flag == 0:
            break


def test10(a):
    n = len(a)
    for i in range(n):
        indx = 0
        for j in range(n - i - 1):
            if a[j + 1] > a[indx]:
                indx = j + 1
        a[indx], a[-(i + 1)] = a[-(i + 1)], a[indx]
    return a


def test11(a, init, gap):
    for j in range(init, len(a), gap):
        now = a[j]
        indx = j
        while indx >= gap and a[indx - gap] > now:
            a[indx] = a[indx - gap]
            indx = indx - gap
        a[indx] = now
    return a


def test12(a):
    count = len(a) // 2
    while count > 0:
        for i in range(count):
            a = test11(a, i, count)
        count //= 2
    return a


def test13(a):
    if len(a) > 1:
        count_h = len(a) // 2
        left_list = a[:count_h]
        right_list = a[count_h:]
        test13(left_list)
        test13(right_list)
        # 总
        k = 0
        # 左右
        i = 0
        j = 0
        while i < len(left_list) and j < len(right_list):
            if left_list[i] > right_list[j]:
                a[k] = right_list[j]
                j += 1
            else:
                a[k] = left_list[i]
                i += 1
            k += 1
        # 缺的得放回去
        while i < len(left_list):
            a[k] = left_list[i]
            i += 1
            k += 1
        while j < len(right_list):
            a[k] = right_list[j]
            j += 1
            k += 1
        print(a)


def test14(a, first, end):
    pivotvalue = a[first]
    left = first + 1
    right = end
    flag = 0
    while flag == 0:
        # 经典双指针
        if a[left] <= pivotvalue and left <= right:
            left += 1
        if a[right] >= pivotvalue and right >= left:
            right -= 1

        if right < left:
            flag = 1
        else:
            # 左右交换
            a[left], a[right] = a[right], a[left]
    # 跟参考点交换
    a[first], a[right] = a[right], a[first]
    return right


def test15(a, first, last):
    if first < last:
        rightx = test14(a, first, last)
        test15(a, first, rightx - 1)
        test15(a, rightx + 1, last)


a = [54, 26, 93, 17, 77, 31, 44, 55, 20]
test15(a, 0, len(a) - 1)
print(1)
'''
a = test1("( 1 + 2 ) * ( 3 + 4 )")
b = test2(a)
print(test3([1, 5, 10, 21, 25], 64))
a = np.arange(99)
a = np.delete(a, 56)
ax=test4(a, 57)
print(hashStr('cat',11))
a = [54, 26, 93, 17, 77, 31, 44, 55, 20]
test5(a, 11)
test6(a, 11)
test7(a, 11)
test8(a, 11)
test9(a)
test10(a)
test11(a,0,1)
print(test12(a))
test13(a)
'''
