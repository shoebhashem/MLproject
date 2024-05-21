import os
import re

def replace_in_file(file_path, old_str, new_str):
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    # Perform the replacement
    new_content = re.sub(old_str, new_str, file_content)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)

def traverse_and_replace(directory, old_str, new_str):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                replace_in_file(file_path, old_str, new_str)

# Replace occurrences of 'carPricePredictor' with 'carPricePredictor' in all files
traverse_and_replace('.', 'carPricePredictor', 'carPricePredictor')
