

import re
import json



# Regular expression to capture the relevant fields from each log line
# Explanation:
# ^Step: (\d+)                     - Matches "Step: " at the start and captures the step number (integer)
# , Loss: (\d+\.\d+)              - Matches ", Loss: " and captures the total loss (float)
#  \(Seg: (\d+\.\d+)               - Matches " (Seg: " (escaped parenthesis) and captures Seg loss (float)
# , Score: (\d+\.\d+)\)            - Matches ", Score: ", captures Score loss (float), and matches closing parenthesis ")"
# , Batch IoU: (\d+\.\d+)         - Matches ", Batch IoU: " and captures Batch IoU (float)
# , Mean IoU: (\d+\.\d+)$         - Matches ", Mean IoU: ", captures Mean IoU (float), and ensures it's at the end of the line
pattern = re.compile(
    r"^Step: (\d+), Loss: (\d+\.\d+) \(Seg: (\d+\.\d+), Score: (\d+\.\d+)\), Batch IoU: (\d+\.\d+), Mean IoU: (\d+\.\d+)$"
)

parsed_data = []

with open('new.txt', 'r') as file:
    # Process each line in the log text
    for line in file:

        line = line.strip() 
        if line.startswith("Model saved"):
            # Skip lines that start with "Model saved"
            continue
        match = pattern.match(line)
        if match:
            # If a line matches the pattern, extract the captured groups
            step, loss, seg_loss, score_loss, batch_iou, mean_iou = match.groups()

            # Create a dictionary for the current step's data
            data_point = {
                "Step": int(step),
                "Loss": float(loss),
                "Seg_Loss": float(seg_loss),
                "Score_Loss": float(score_loss),
                "Batch_IoU": float(batch_iou),
                "Mean_IoU": float(mean_iou)
            }
            parsed_data.append(data_point)
        # Lines like "Model saved..." will not match and will be ignored

# Convert the list of dictionaries to a JSON string with indentation
json_output = json.dumps(parsed_data, indent=4)

# Print the resulting JSON
print(json_output)

with open('parsed_data.json', 'w') as json_file:
    # Write the JSON string to a file
    json_file.write(json_output)