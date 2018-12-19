import os,re
import json,glob    

data = []
with open('labels.json') as f:
	for line in f:
		data.append(json.loads(line))


dataset_file = open('dataset.lst','w')
images = os.listdir('./DATASET/images')

# sort string as numbers 
def atoi(text):
	return int(text) if text.isdigit() else text
def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)',text) ]
 
images = sorted(images, key=natural_keys)
print(images)


for img in range(len(images)):
	data_ = [0 if x==-1 else x for x in data[img]]
	labels = '\t'.join(map(str, data_))
	write = [str(img+1),labels,'images/'+images[img],'\n']
	dataset_file.write('\t'.join(write))

dataset_file.close()
