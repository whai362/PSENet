import torch

EPS = 1e-6

def acc_single(a, b, mask):
    ind = mask == 1
    if torch.sum(ind) == 0:
        return 0
    correct = (a[ind] == b[ind]).float()
    acc = torch.sum(correct) / correct.size(0)
    return acc

def acc(a, b, mask, reduce=True):
    batch_size = a.size(0)

    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    acc = a.new_zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        acc[i] = acc_single(a[i], b[i], mask[i])

    if reduce:
        acc = torch.mean(acc)
    return acc