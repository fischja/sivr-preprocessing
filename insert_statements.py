from pathlib import Path
import pandas as pd
import re
import ast
import numpy as np


df = pd.read_csv(f'./openimages.names').reset_index()
df = df.rename(columns={'index': 'OpenImagesConceptId'})
df['OpenImagesConceptName'] = df['OpenImagesConceptName'].str.upper()
df['OpenImagesConceptId'] = df['OpenImagesConceptId'] + 1000
df.to_csv(f'./db_data/data_openimagesconceptname.csv', index=False)

"""
df = pd.read_csv(f'./yolo_preds.csv', index_col='Unnamed: 0')
df['V3CId'] = df.index.str[4:9].astype('int16')

keyframe_nums = []
for i in df.index:
    keyframe_nums.append(int(re.findall("_(.*?)_", i)[0]))
df['KeyframeNumber'] = keyframe_nums
df['KeyframeNumber'] = df['KeyframeNumber'].astype('int16')

df = df.melt(id_vars=['V3CId', 'KeyframeNumber'], var_name='OpenImagesConceptId', value_name='Score')
df['OpenImagesConceptId'] = df['OpenImagesConceptId'].astype('int16') + 1000
df = df.loc[df['Score'] > 0.0, :]

print(df.info())
print(df.head())

df.to_csv(f'./db_data/data_openimagesconceptscore.csv', index=False)



df = pd.read_csv(f'./color_scores_q256_200.csv', index_col='Unnamed: 0')
df['V3CId'] = df.index.str[4:9].astype('int16')
df['KeyframeNumber'] = df.index.str.split("_").map(lambda x: x[1]).astype('int16')
df = df.melt(id_vars=['V3CId', 'KeyframeNumber'], var_name='ColorId', value_name='Score')
df.to_csv(f'./db_data/data_colorscore.csv', index=False)
print(df.head())
print(df.info())


df = pd.read_csv(f'./imagenet_mean_preds.csv', index_col='Unnamed: 0')
df['V3CId'] = df.index.str[4:9].astype('int16')

print(df.loc[:, 'l_0'])

keyframe_nums = []
for i in df.index:
    keyframe_nums.append(int(re.findall("_(.*?)_", i)[0]))
df['KeyframeNumber'] = keyframe_nums
df['KeyframeNumber'] = df['KeyframeNumber'].astype('int16')

df = df.melt(id_vars=['V3CId', 'KeyframeNumber'], var_name='ImageNetConceptId', value_name='Score')
df['ImageNetConceptId'] = df['ImageNetConceptId'].str[2:].astype('int16')
print(df.info())
print(df.head())
df.to_csv(f'./db_data/data_imagenetconceptscore.csv', index=False)


p = Path(r'F:\\UZH\\Interactive Video Retrieval\\keyframes\\keyframes')

v3c_ids = []
for img_path in p.iterdir():
    v3c_ids.append(int(img_path.stem))

df = pd.DataFrame({'V3CId': v3c_ids})
df.to_csv(f'./db_data/data_video.csv', index=False)
print(df.head())


v3c_ids = []
keyframe_nums = []
for img_path in p.glob('**/*.png'):
    v3c_id = int(img_path.stem[4:9])
    keyframe_num = int(re.findall("_(.*?)_", img_path.stem)[0])
    v3c_ids.append(v3c_id)
    keyframe_nums.append(keyframe_num)

df = pd.DataFrame({'V3CId': v3c_ids, 'KeyframeNumber': keyframe_nums})
print(df.head())
df.to_csv(f'./db_data/data_keyframe.csv', index=False)


ids = list(range(1000))
df = pd.DataFrame({'ImageNetConceptId': ids})
print(df.head())
df.to_csv(f'./db_data/data_imagenetconcept.csv', index=False)

with open('./imagenet_labels.txt', 'r') as f:
    d = f.read()

d = ast.literal_eval(d)
concept_ids, concept_names = [], []
for key in d:
    names = [x.strip() for x in d[key].lower().split(',')]
    for name in names:
        if name in concept_names:
            continue
        concept_ids.append(key)
        concept_names.append(name)

df = pd.DataFrame({'ImageNetConceptId': concept_ids, 'Name': concept_names})
print(df.head())
df.to_csv(f'./db_data/data_imagenetconceptname.csv', index=False)
"""
