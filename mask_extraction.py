import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.io import decode_image
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

weights = FCN_ResNet50_Weights.DEFAULT
transforms = weights.transforms(resize_size=None)

model = fcn_resnet50(weights=weights, progress=False)
model = model.eval()

image = Image.open("source.jpg").convert("RGB")
to_tensor = ToTensor()
image = to_tensor(image)
image = torch.stack([transforms(image)])

result = model([image])['out']

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

normalized_masks = torch.nn.functional.softmax(result, dim=1)
boolean_target_masks = (normalized_masks.argmax(1) == sem_class_to_idx['aeroplane'])

plt.imshow(boolean_target_masks[0].detach().numpy(), cmap='gray')
plt.show()
plt.imsave("mask.png", boolean_target_masks[0].detach().numpy(), cmap='gray')