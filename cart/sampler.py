from random import Random
from typing import *
import math


class Sampler:

    def __init__(self, seed=42, *args, **kwargs):
        self.seed = seed
        self.random = Random(self.seed)

    def sample(self, x: Iterable):
        raise NotImplemented


class SimpleSampler(Sampler):

    def __init__(self, seed=42, quantity='all'):
        super(SimpleSampler,self).__init__(seed=seed, quantity=quantity)

        is_supported_str = quantity in ['all', 'sqrt', 'log2', 'auto', None]
        is_supported_int = isinstance(quantity, int) and 0 <= quantity
        is_supported_float = isinstance(quantity, float) and 0.0 <= quantity <= 1.0

        assert is_supported_str or is_supported_int or is_supported_float, "unsupported quantity value:%s"%str(quantity)

        self.quantity = quantity

    def sample(self, x: Iterable):

        if self.quantity == 'all' or self.quantity is None:
            return x

        elif self.quantity in ['sqrt', 'auto']:
            k = int(math.sqrt(len(x)))

        elif self.quantity == 'log2':
            k = int(math.log2(len(x)))

        elif isinstance(self.quantity, int):
            assert 0 <= self.quantity <= len(x), \
                "quantity should be in range 0 <= quantity <= population (quantity=%d,population=%d)"%(self.quantity,len(x))
            k = self.quantity

        elif isinstance(self.quantity, float):
            assert 0.0 <= self.quantity <= 1.0, \
                "quantity should be in range 0.0 <= quantity <= 1.0 (quantity=%f)" % self.quantity
            k = int(self.quantity * len(x))

        else:
            assert False, "unsupported quantity value: %s"%str(self.quantity)

        return self.random.sample(x, k)


class BootstrapSampler(Sampler):
    def __init__(self, seed=42, quantity='all'):
        super(BootstrapSampler, self).__init__(seed=seed, quantity=quantity)

        is_supported_str = quantity in ['all', None]
        is_supported_int = isinstance(quantity, int) and 0 <= quantity
        is_supported_float = isinstance(quantity, float) and 0.0 <= quantity

        assert is_supported_str or is_supported_int or is_supported_float, "unsupported quantity value:%s" % str(
            quantity)

        self.quantity = quantity

    def sample(self, x: Iterable):

        if self.quantity == 'all' or self.quantity is None:
            return x

        elif isinstance(self.quantity, int):
            assert 0 <= self.quantity, \
                "quantity should be in range 0 <= quantity (quantity=%d,population=%d)" % (self.quantity, len(x))
            k = self.quantity

        elif isinstance(self.quantity, float):
            assert 0.0 <= self.quantity, \
                "quantity should be in range 0.0 <= quantity (quantity=%f)" % self.quantity
            k = int(self.quantity * len(x))

        else:
            assert False, "unsupported quantity value: %s" % str(self.quantity)

        return [self.random.sample(x,1)[0] for i in range(k)]
