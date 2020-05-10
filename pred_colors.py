from PIL import Image, ImageColor
from pathlib import Path
from collections import Counter
from skimage.color import rgb2lab, deltaE_ciede2000
import pandas as pd
import numpy as np


base_colors_hex = [
    '#000000',
    '#8b4513',
    '#006400',
    '#778899',
    '#000080',
    '#ff0000',
    '#ffa500',
    '#ffff00',
    '#c71585',
    '#00ff00',
    '#00fa9a',
    '#00ffff',
    '#0000ff',
    '#ff00ff',
    '#1e90ff',
    '#eee8aa',
]

base_colors_rgb = {h: ImageColor.getrgb(h) for h in base_colors_hex}
base_colors_lab = {h: rgb2lab([[[base_colors_rgb[h][0],
                                 base_colors_rgb[h][1],
                                 base_colors_rgb[h][2]]]]) for h in base_colors_hex}

p = Path(r'D:\Interactive Video Retrieval\thumbnails\thumbnails')
res_dict = {}
n_keyframes = 108645
counter = 0
for img_path in p.glob('**/*.png'):
    counter += 1
    print(counter, '/', n_keyframes)
    img = Image.open(img_path)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    palette = np.array(img.getpalette()).reshape(256, 3)
    # img = img.resize((600, 400), Image.LANCZOS)

    colors = Counter(img.getdata())
    n_pixels = img.size[0] * img.size[1]

    n_most_common = min(len(colors), 200)
    color_counts = [(palette[c[0]], c[1]) for c in colors.most_common(n_most_common)]

    color_counts = [(rgb2lab([[[c[0][0], c[0][1], c[0][2]]]]),
                     (c[1] / n_pixels) * 1e6)
                    for c in color_counts]

    scores = {}
    for h in base_colors_lab:
        scores[h] = sum([deltaE_ciede2000(base_colors_lab[h], c[0]) * c[1] for c in color_counts])[0][0]

    res_dict[img_path.stem] = scores

df = pd.DataFrame.from_dict(res_dict, orient="index")
df.to_csv(f'./color_scores_q256_150.csv')
print(df)
