import nbformat as nbf
import os

chapter_num = 4
chp = f'{chapter_num:02d}'
file_path = f'C:\\Users\\eduar\\mydevrepo\\ml_sebastian\\machine-learning-book_repo\\ch{chp}\\ch{chp}.ipynb'
basename = os.path.basename(file_path)

new_path = r'C:\Users\eduar\mydevrepo\ml_sebastian\dev'
new_file = os.path.join(new_path, basename[:-6] + '_dev.ipynb')

ntbk = nbf.read(file_path, nbf.NO_CONVERT)

cells = []

for cell in ntbk.cells:
    if cell.cell_type == "code":
        if cell.source.startswith(" "):
            cell.source = cell.source.lstrip()
        if cell.source.startswith("import sys"):
            continue 
        if cell.source.startswith("from python_environment_check"):
            continue

        cell.execution_count = None

        new_code = ''
        if cell.source.startswith("Image(filename="):
            new_code = "# No code"

        code = cell.source.split('\n')
        comment = False
        for line in code:
            if line.startswith('#plt.savefig'):
                continue
            if line.startswith('# plt.savefig'):
                continue

            if line.startswith('#'):
                new_code += line + '\n'

            short_line = line.strip()
            if short_line.startswith('"""'):
                new_code += line + '\n'
                if short_line.endswith('"""') and len(short_line) > 3:
                    continue
                comment = True if not comment else False
                continue
            
            if comment:
                new_code += line + '\n'

        cell.source = new_code
            

    if cell.cell_type == "markdown":
        if cell.source.startswith("## Package version checks"):
            continue
        elif cell.source.startswith("Add folder to path in order"):
            continue
        elif cell.source.startswith("Check recommended package versions:"):
            continue

    cells.append(cell)

ntbk.cells = cells

nbf.write(ntbk, new_file, version=nbf.NO_CONVERT)
