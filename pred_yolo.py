import numpy as np
import cv2
import os
from pathlib import Path
import pandas as pd


def get_preds(net, img_path):
	# load our input image and grab its spatial dimensions
	image = cv2.imread(img_path)
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image,
								 1 / 255.0,
								 (416, 416),
								 swapRB=True,
								 crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	scores = None
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores_new = detection[5:]
			scores = scores_new if scores is None else scores + scores_new

	return scores


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.abspath('./openimages.names')
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.abspath('./yolov3-openimages.weights')
configPath = os.path.abspath('./yolov3-openimages.cfg')

# load our YOLO object detector
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

p = Path(r'D:\Interactive Video Retrieval\keyframes\keyframes')

res_dict = {}
n_keyframes = 108645
counter = 0

for img_path in p.glob('**/*.png'):
	counter += 1
	print(counter, '/', n_keyframes)
	res = get_preds(net=net, img_path=str(img_path.resolve()))
	res_dict[img_path.stem] = res.reshape(-1, )

df = pd.DataFrame.from_dict(res_dict, orient="index")
df.to_csv(f'./yolo_preds.csv')
print(df)
