from feature_extraction import create_graph, extract_features
from SVM import train_svm_classifer
from time import gmtime, strftime
import os
import re

model_dir = '/home/samsony/ImageNet/inception_dec_2015/tensorflow_inception_graph.pb'
image_dir = '/home/samsony/ImageNet/Data/'
model_out = 'model.pkl'

lst = []
img_list = []
labels = []

for l in os.listdir(image_dir):
	lst.append(eval(l))
lst.sort()

for l in lst:
	file_list = os.listdir(image_dir + str(l))
	file_list = sorted(file_list, key=lambda x: (int(re.sub('\D','',x)),x))
	for f in file_list:
		path = os.path.join(image_dir, str(l), f)
		img_list.append(path)
		labels.append(l)

target = open('log.txt', 'w')
target.write("Create Graph Now\n")
target.flush()
target.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
create_graph(model_dir)
target.write("Done Create Graph\n")
target.flush()
target.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
features = extract_features(img_list, True)
target.write("Done creating features\n")
target.flush()
target.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
train_svm_classifer(features, labels, model_out)
target.write("Done training\n")
target.flush()
target.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
target.close()


