import io
import pandas as pd
from PIL import Image
from openpyxl import load_workbook

# Load the workbook using openpyxl to check for embedded images
wb = load_workbook('Blueberries.xlsx', data_only=True)
ws = wb["Sheet1"]

# Also load with pandas to get the berry counts
df = pd.read_excel('Blueberries.xlsx')

# Extract and list all images in the worksheet
print("Images found:", len(ws._images))

# Extract images from the worksheet
images = ws._images

# Save extracted images into an array
image_data = []
for img in images:
    # Convert image to byte stream and then load it as a PIL image
    img_bytes = io.BytesIO(img._data())
    pil_image = Image.open(img_bytes)
    image_data.append(pil_image)

# Combine extracted images with the blueberry count column
berry_counts = df['Blueberries'].dropna().tolist()

# Ensure both lists have the same length before pairing them
image_berry_array = list(zip(image_data[:len(berry_counts)], berry_counts))

# Display the first few entries
print(f"Created array with {len(image_berry_array)} image-berry count pairs")
# Can't directly print PIL images, so just show the count
print(f"First 5 entries (showing only count values):")
for i, (img, count) in enumerate(image_berry_array[:5]):
    print(f"Entry {i+1}: Image size {img.size}, Berry count: {count}")

# Optionally save the array
# Note: PIL images can't be directly saved with np.save
# You might want to save the images to files instead