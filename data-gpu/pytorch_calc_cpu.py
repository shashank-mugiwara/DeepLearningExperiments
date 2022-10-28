import torch
import timeit

x = torch.rand(2 ** 11, 2 ** 11)
num_of_rep = 1500
cpu_time = timeit.timeit("x@x", globals=globals(), number=num_of_rep)
print("Time taken to complete {} number of self-multiply operations: {}".format(num_of_rep, cpu_time))
