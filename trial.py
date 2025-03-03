import tifffile as tiff
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

file = "/gpfs/home/rs9522/cyto-mask/Composite_with_masks.tif"
img = tiff.imread(file)
print(img.shape)

bf = img[:, 1, :, :]
mask = img[:,3, :, :]
print(bf.shape)
print(mask.shape)


# train test split

X_train, X_test, y_train, y_test = train_test_split(bf, mask, test_size=0.2, random_state=22)

# convert the np arrays to tensors

X_train_tensor = torch.tensor(X_train /255.0, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test /255.0, dtype= torch.float32)
y_train_tensor = torch.tensor(y_train, dtype= torch.float32)
y_test_tensor = torch.tensor(y_test, dtype= torch.float32)

print("size after rescale: ",  X_train_tensor.shape)

# not make a tensor dataset

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoader function is similar to the ImageDataGenerator in tensorflow
train_loader = DataLoader(train_data, batch_size = 5, shuffle =True)
test_loader = DataLoader(test_data, batch_size = 5, shuffle = True)

bf, mask = next(iter(train_loader))

print(img.shape)
print(mask.shape)

# sample bf image
plt.imshow(X_train_tensor[0].cpu().numpy(), cmap= "gray")
plt.axis("off")
plt.savefig("test_img_bf.png")
plt.show()

# sample mask img

plt.imshow(y_train_tensor[0].cpu().numpy(), cmap= "gray")
plt.axis("off")
plt.savefig("test_img_mask.png")
plt.show()