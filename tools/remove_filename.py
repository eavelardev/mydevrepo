import os

path = ""
filename = ""

for root, _, files in os.walk(path):
    for name in files:
        if name.endswith(filename):
            file_path = os.path.join(root, name)
            # Check files to remove
            print(file_path)
            # os.remove(file_path)
