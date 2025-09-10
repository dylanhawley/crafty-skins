from pathlib import Path
from PIL import Image
import shutil

SRC = Path('/Users/dylanhawley/Projects/crafty-skins/data/raw/skins/')
DST = Path('/Users/dylanhawley/Projects/crafty-skins/data/raw/skins-post1.8/')
DST.mkdir(parents=True, exist_ok=True)

for img in SRC.iterdir():
    if not img.is_file():
        continue
    try:
        with Image.open(img) as im:
            if im.size == (64, 64):
                shutil.copy2(img, DST / img.name)
                print(f"Copied {img.name}")
    except OSError:
        # not an image
        pass