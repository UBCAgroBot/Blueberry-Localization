import matplotlib.pyplot as plt
from torchvision.io import read_image

image = read_image("./data/PNGImages/FudanPed00046.png")
mask = read_image("./data/PedMasks/FudanPed00046_mask.png")

# specifies the plot size (16 inches width, 8 inches height)
plt.figure(figsize=(16, 8))
# specifies that the plot has 1 row and 2 columns, and that 
# the following subplot will occupy the first index of that plot
# (i.e., row 1, column 1)
plt.subplot(121)
plt.title("Image")
# Permutes the tensor's dimensions
# i.e., if we had (CC, H, W)
# i.e., now we have (H, W, CC)
plt.imshow(image.permute(1,2,0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1,2,0))
plt.show()

