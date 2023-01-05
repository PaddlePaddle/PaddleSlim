def sum_list(a, j):
    b = 0
    for i in range(len(a)):
        if i != j:
            b += a[i]
    return b
