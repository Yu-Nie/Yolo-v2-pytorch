import os
import cv2
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from src.data_augmentation import *
from src.edge_detection import *


class VOCDataset(Dataset):
    def __init__(self, root_path="data/VOCdevkit", year="2012", mode="train", image_size=448, is_training=True):
        if (mode in ["train", "val", "trainval", "test"] and year == "2007") or (
                mode in ["train", "val", "trainval"] and year == "2012"):
            self.data_path = os.path.join(root_path, "VOC{}".format(year))
        id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))
        self.ids = [id.strip() for id in open(id_list_path)]
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        image_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(id))
        annot = ET.parse(image_xml_path)

        segment_path = os.path.join(self.data_path, "SegmentationObject", "{}.png".format(id))

        objects = []
        ratio_list = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                                  ["xmin", "xmax", "ymin", "ymax"]]
            mask_ratio_text = obj.find('bndbox').find('mask_ratio').text
            mask_ratio = [float(r) for r in mask_ratio_text.split(',')]
            ratio_list.append(mask_ratio)

            # object_bbox = image[ymin:ymax, xmin:xmax, :]
            '''# generate mask_ratio in xml files
            if os.path.isfile(segment_path):
                ratio = get_ratio(segment_path, xmin, xmax, ymin, ymax)

                mask_name = ET.Element('mask_ratio')
                mask_name.text = str(ratio)[1:-1]
                bbox = obj.find('bndbox')
                bbox.append(mask_name)
                root = annot.getroot()
                pretty_xml(root, '\t', '\n')
                new_path = image_xml_path.replace('Annotations-old', 'Annotations')
                annot.write(new_path)
            '''

            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])

        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))
        i = 0
        for item in objects:
            for j in range(16):
                item.append(ratio_list[i][j])
            i += 1
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
