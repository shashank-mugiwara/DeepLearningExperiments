import timeit
import torch

x = torch.rand(2**11, 2**11)
print("Is CUDA available: {}".format(torch.cuda.is_available()))

device = torch.device("cuda")
x = x.to(device)

num_of_rep = 1500
gpu_time = timeit.timeit("x@x", globals=globals(), number=num_of_rep)
print("Time taken to complete {} number of self-multiply operations: {}".format(num_of_rep, gpu_time))