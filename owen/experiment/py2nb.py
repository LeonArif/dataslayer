"""Convert a '# %%' cell-marked script into a Jupyter notebook: python py2nb.py puzzle_take.py puzzle_take.ipynb"""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf8").read().splitlines()
cells, cur, kind = [], [], None

def flush():
    body = "\n".join(cur).strip("\n")
    if kind and body:
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.splitlines(True)})

for l in lines:
    if l.startswith("# %%"):
        flush(); cur = []
        kind = "markdown" if "[markdown]" in l else "code"
    elif l.strip() == "# STAGE_A_END":
        continue
    else:
        cur.append(l)
flush()

# Kaggle: make sure a transformers version with ViTPose is installed (needs Internet ON)
install = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
           "source": ["# Kaggle: Settings -> Accelerator GPU (T4/P100), Internet ON\n",
                      "import importlib.util, subprocess, sys\n",
                      "try:\n",
                      "    from transformers import VitPoseForPoseEstimation  # noqa\n",
                      "except Exception:\n",
                      "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers>=4.50'], check=True)\n"]}
cells.insert(1, install)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(dst, "w", encoding="utf8"), indent=1)
print(f"{len(cells)} cells -> {dst}")
