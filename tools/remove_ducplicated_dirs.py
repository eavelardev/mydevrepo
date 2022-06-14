import os
from pathlib import Path

def get_size(path_name):
    path = Path(path_name)
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

path_with_folders_to_keep = r""
path_with_folders_to_remove = r""

dirs_keep = {}
dirs_remove = []

for root, _, files in os.walk(path_with_folders_to_keep):
    folder = root[2:]
    dirs_keep[folder] = get_size(root)


for root, _, files in os.walk(path_with_folders_to_remove):
    folder = root[29:]
    if dirs_keep[folder] != get_size(root):
        dirs_remove.append(folder)

for dir in dirs_remove:
    print(dir)
