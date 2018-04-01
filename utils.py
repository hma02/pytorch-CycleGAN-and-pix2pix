import torch
from torch.autograd import Variable
import os


def to_var(x, requires_grad=False, volatile=False):
    x = Variable(x, requires_grad=requires_grad, volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def one_hot(x, n_classes):
    if isinstance(x, Variable):
        is_var = True
        x = x.data
    else:
        is_var = False
    x_one_hot = torch.zeros(x.size(0), n_classes)
    if torch.cuda.is_available():
        x_one_hot = x_one_hot.cuda()
    x_one_hot.scatter_(1, x.unsqueeze(1), 1)
    if is_var:
        x_one_hot = Variable(x_one_hot)
    return x_one_hot