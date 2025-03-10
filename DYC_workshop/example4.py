

result = []
for x in range(10):
    if x % 2 != 1 & x != 4:
        if x % 2 == 0:
            result.append(x*2)
        else:
            result.append(x*3)


def is_empty(lst):
    return len(lst) == 0