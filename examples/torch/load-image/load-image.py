import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)
import imageio.v2 as imageio

img_arr = imageio.imread('./data/puppy.jpg')
print(img_arr.shape)

img = torch.from_numpy(img_arr)
img = img.permute(2, 0, 1)
img = img[:3]
print(img)

batch = img
batch = batch.float()
batch /= 255.0
print(batch)
print(batch.shape)