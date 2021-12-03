import torch
import torch.nn as nn
import torchvision.transforms as T

"""
This file shows an example of transforms using nn.sequential which can be done on the GPU, placed in the main for loop during training which is faster than transforms.Compose
"""

# t_mean, t_std = get_mean_std("train")
t_mean, t_std = torch.zeros(1, 3), torch.ones(1, 3)

t_mean = (t_mean).tolist()
t_std = (t_std).tolist()

print("mean values:", t_mean)
print("std values:", t_std)

val_transform = nn.Sequential(
    T.Normalize(mean=t_mean, std=t_std),
)
test_transform = nn.Sequential(
    T.Normalize(mean=t_mean, std=t_std),
)

train_transform = nn.Sequential(
    T.Normalize(mean=t_mean, std=t_std),
    T.RandomRotation(degrees=45),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.05),
    T.RandomGrayscale(p=0.2),
    T.RandomErasing(p=0.5),
)
#     T.ColorJitter(brightness=(0, 0.5), hue=(-0.3, 0.3)),
#     T.RandomAffine(degrees=(-45, 45), translate=(0, 0.2), scale=(0.5, 1.0)),
