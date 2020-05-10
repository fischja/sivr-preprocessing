import pandas as pd


df1 = pd.read_csv('./imagenet_inceptionresnetv2_preds.csv').set_index('Unnamed: 0')
df2 = pd.read_csv('./imagenet_inceptionv3_preds.csv').set_index('Unnamed: 0')
df3 = pd.read_csv('./imagenet_vgg16_preds.csv').set_index('Unnamed: 0')

df = pd.concat([df1, df2, df3]).groupby(level=0).mean()
df.columns = ['l_' + str(c) for c in df.columns]

df.to_csv(f'./imagenet_mean_preds.csv')
