import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('/home/suriya/cyto-mask/sam2/notebooks/parsed_data.json', 'r') as file:
    data = json.load(file)

# Extract relevant data
steps = [entry['Step'] for entry in data if 'Step' in entry]
loss = [entry['Loss'] for entry in data if 'Loss' in entry]
mean_iou = [entry['Mean_IoU'] for entry in data if 'Mean_IoU' in entry]

# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(steps, loss, label='Loss', color='blue')
plt.title('Loss Curve')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy (Mean IoU) Curve
plt.figure(figsize=(10, 5))
plt.plot(steps, mean_iou, label='Mean IoU', color='green')
plt.title('Accuracy Curve (Mean IoU)')
plt.xlabel('Steps')
plt.ylabel('Mean IoU')
plt.legend()
plt.grid(True)
plt.show()