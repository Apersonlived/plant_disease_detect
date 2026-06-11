import json, glob

for path in glob.glob("*.ipynb"):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed: {path}")