import keras.applications.vgg16 as vgg16
import keras.applications.inception_resnet_v2 as inception_resnet_v2
import keras.applications.inception_v3 as inception_v3
from keras.preprocessing import image
import numpy as np
from pathlib import Path
import pandas as pd


def get_preds(model, preprocessor, decoder, target_size, img_path):
    img = image.load_img(img_path, target_size=target_size, interpolation='lanczos')

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocessor(x)

    preds = model.predict(x)

    # print('Predicted:', decoder(preds, top=3)[0])
    # print(preds.reshape(-1, ).argsort()[-3:][::-1])

    return preds


models = {
    'vgg16': vgg16.VGG16(weights='imagenet'),
    'inceptionresnetv2': inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'inceptionv3': inception_v3.InceptionV3(weights='imagenet'),
}

preprocessors = {
    'vgg16': vgg16.preprocess_input,
    'inceptionresnetv2': inception_resnet_v2.preprocess_input,
    'inceptionv3': inception_v3.preprocess_input,
}

decoders = {
    'vgg16': vgg16.decode_predictions,
    'inceptionresnetv2': inception_resnet_v2.decode_predictions,
    'inceptionv3': inception_v3.decode_predictions,
}

target_sizes = {
    'vgg16': (224, 224),
    'inceptionresnetv2': (299, 299),
    'inceptionv3': (299, 299),
}

p = Path(r'F:\UZH\Interactive Video Retrieval\keyframes\keyframes')

for model_name in models:
    res_dict = {}
    for img_path in p.glob('**/*.png'):
        print(img_path.stem)
        res = get_preds(model=models[model_name],
                        preprocessor=preprocessors[model_name],
                        decoder=decoders[model_name],
                        target_size=target_sizes[model_name],
                        img_path=img_path)
        res_dict[img_path.stem] = res.reshape(-1, )

    df = pd.DataFrame.from_dict(res_dict, orient="index")
    df.to_csv(f'./imagenet_{model_name}_preds.csv')
    print(df)
