import os
import shutil

segment_dir = 'data/VOCdevkit/VOC2012/SegmentationObject'
image_dir = 'data/VOCdevkit/VOC2012/JPEGImages'
annot_dir = 'data/VOCdevkit/VOC2012/Annotations'
set_dir = 'data/VOCdevkit/VOC2012/ImageSets'

id_list = []
for image in os.listdir(segment_dir):
    id = image.split('.')[0]
    id_list.append(id)
    # image_name = id + '.jpg'
    # shutil.copy(os.path.join(image_dir, image_name), 'data_new/VOCdevkit/VOC2012/JPEGImages/' + image_name)
print(len(id_list))

for dirc in os.listdir(annot_dir):
    idd = dirc.split('.')[0]
    if idd in id_list:
        dirct = os.path.join(annot_dir, dirc)
        shutil.copy(dirct, 'Annotations/' + dirc)
    '''
    with open(os.path.join(dirct, file), 'r') as re, open(os.path.join(new_file, file), 'w') as wr:
        for index, line in enumerate(re):
            if line.split(' ')[0].strip('\n') in id_list:
                wr.write(line)
    '''