import torch

a = torch.ones(3)
print(a)

a[2] = 2.0
print(a)

points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
print(points)

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)

print(points[0])

points_storage = points.storage()
print(points_storage)
print(points_storage[0])

second_point = points[1]
print(second_point.storage_offset())
print(second_point.stride())

print(points.t())

points_gpu = points.to(device='cuda')
print(points_gpu)
