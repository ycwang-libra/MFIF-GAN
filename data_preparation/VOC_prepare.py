import os
class_path = '~/dataset/VOC2012/VOC2012trainval/VOCdevkit/VOC2012/SegmentationClass'
input0_folder  = class_path

a = []   # all of name in SegmentationClass
for root, dirs, files in os.walk(input0_folder):
    for filename in (x for x in files if x.endswith('.png')):
        filepath = os.path.join(root, filename)
        
        object_class = filename.split('.')[0]
        a.append(object_class)
        
del class_path, input0_folder, root, dirs, files, filename, filepath, object_class 

###############   dataset collection  ######################
'''
extract ''image_jpg'' from ''JPEGImage'' corresponding to ''SegmentationClass''
'''
from PIL import Image

image_path = '~/dataset/VOC2012/VOC2012trainval/VOCdevkit/VOC2012/JPEGImages'

input1_folder = image_path
output_folder = '~/dataset/multi_focus_dataset_by_VOC2012/image_jpg'
os.mkdir(output_folder)
        
for i in a:
    old_path = input1_folder + "\\" + str(i) + '.jpg'
    new_path = output_folder + "\\" + str(i) + '.jpg'
    img = Image.open(old_path)
    img.save(new_path)

#reset