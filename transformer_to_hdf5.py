import os
from os.path import join, isfile
import numpy as np
import h5py
from glob import glob
from sklearn import model_selection
import torch,torchfile
# from torch.utils.serialization import load_lua
from PIL import Image
import yaml
import io
import pdb
import transform

with open('config.yaml', 'r') as f:
	config = yaml.load(f,yaml.FullLoader)

images_path = config['flowers_images_path']
embedding_path = config['flowers_embedding_path']
text_path = config['flowers_text_path']
datasetDir = config['flowers_dataset_path']

val_classes = open(config['flowers_val_split_path']).read().splitlines()
train_classes = open(config['flowers_train_split_path']).read().splitlines()
test_classes = open(config['flowers_test_split_path']).read().splitlines()

# print(val_classes)
f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

model = transform.load_model()
for _class in sorted(os.listdir(embedding_path)):
	# print(_class)
	split = ''
	if _class in train_classes:
		split = train
	elif _class in val_classes:
		split = valid
	elif _class in test_classes:
		split = test

	data_path = os.path.join(embedding_path, _class)
	# data_path = data_path[:-2]
	txt_path = os.path.join(text_path, _class)
	# txt_path = txt_path[:-2]
	# print(data_path,txt_path)


# Data/images/class_00102.t7/*.t7
# Data/flowers/text_c10/class_00102.t7/*.txt
	for example, txt_file in zip(sorted(glob(data_path + "/.t7")), sorted(glob(txt_path + "/.txt"))):
		# print(example)
		example_data = torchfile.load(example)
		# print(torchfile.T7Reader())
		# print(example_data)
		img_path = str(example_data[b'img'])[2:-1]
		# print(img_path) 
		# embeddings = example_data[b'txt']
		
		# print('before',embeddings)
		example_name = img_path.split('/')[-1][:-4]		
		# print(example_name)
		
		f = open(txt_file, "r")
		txt = f.readlines()
		f.close()
		
		
		img_path = os.path.join(images_path, img_path)
		img = open(img_path, 'rb').read()

		txt_choice = np.random.choice(range(10), 5)

		
		# print('after',embeddings)
		txt = np.array(txt)
		txt = txt[txt_choice]
		# print('txt',txt.astype(object))
		# embeddings = embeddings[txt_choice]
		embeddings = transform.Encode(model,txt)
		# print('after',embeddings)
		dt = h5py.special_dtype(vlen=str)

		for c, e in enumerate(embeddings):
			ex = split.create_group(example_name + '_' + str(c))
			ex.create_dataset('name', data=example_name)
			ex.create_dataset('img', data=np.void(img))
			ex.create_dataset('embeddings', data=e)
			ex.create_dataset('class', data=_class)
			ex.create_dataset('txt', data=txt[c].astype(object), dtype=dt)

		# print(example_name, txt[1], _class)