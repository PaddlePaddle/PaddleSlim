def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    return total_params


def sum_list(a, j):
    b = 0
    for i in range(len(a)):
        if i != j:
            b += a[i]
    return b
