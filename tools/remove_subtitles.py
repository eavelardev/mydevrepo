import os

path = r""
sub_ext = ".srt"
lang_keep = "en"

for root, _, files in os.walk(path):
    for name in files:
        if name.endswith(sub_ext) and not name.endswith(lang_keep + sub_ext):
            file_path = os.path.join(root, name)
            # Check files to remove
            print(file_path)
            # os.remove(file_path)
