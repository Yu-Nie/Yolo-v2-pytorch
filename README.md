# [PYTORCH] YOLO (You Only Look Once)


## How to use my code

With my code, you can:
* **Train your model from scratch**
* **Train your model with my trained model**
* **Evaluate test images with either my trained model or yours**

## Requirements:

* **python 3.6**
* **pytorch 0.4**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
* **numpy**

## Datasets:

I used 1 datases: VOC2012. Statistics of datasets is shown below

| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| VOC2012                |    20   |      5717/13609       |           5823/13841       |



- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:
  ```
  VOCDevkit
  ├── VOC2012
  │   ├── Annotations  
  │   ├── ImageSets
  │   ├── JPEGImages
  │   └── ...
  └── 
  ```

  
## Setting:

* **Model structure**: In compared to the paper, I changed structure of top layers, to make it converge better. You could see the detail of my YoloNet in **src/yolo_net.py**.
* **Data augmentation**: I performed dataset augmentation, to make sure that you could re-trained my model with small dataset (~500 images). Techniques applied here includes HSV adjustment, crop, resize and flip with random probabilities
* **Loss**: The losses for object and non-objects are combined into a single loss in my implementation
* **Optimizer**: I used SGD optimizer and my learning rate schedule is as follows: 

|         Epoches        | Learning rate |
|------------------------|:---------------:|
|          0-4           |      1e-5     |
|          5-79          |      1e-4     |
|          80-109        |      1e-5     |
|          110-end       |      1e-6     |

* In my implementation, in every epoch, the model is saved only when its loss is the lowest one so far. You could also use early stopping, which could be triggered by specifying a positive integer value for parameter **es_patience**, to stop training process when validation loss has not been improved for **es_patience** epoches.



## Training

For each dataset, I provide 2 different pre-trained models, which I trained with corresresponding dataset:
- **whole_model_trained_yolo_xxx**: The whole trained model.
- **only_params_trained_yolo_xxx**: The trained parameters only.

You could specify which trained model file you want to use, by the parameter **pre_trained_model_type**. The parameter **pre_trained_model_path** then is the path to that file.

- **python3 train_voc.py --year year**: For example, python3 train_voc.py --year 2012



## Test

For each type of dataset (VOC or COCO), I provide 3 different test scripts:

If you want to test a trained model with a standard VOC dataset, you could run:
- **python3 test_xxx_dataset.py --year year**: For example, python3 test_coco_dataset.py --year 2014

If you want to test a model with some images, you could put them into the same folder, whose path is **path/to/input/folder**, then run:
- **python3 test_xxx_images.py --input path/to/input/folder --output path/to/output/folder**: For example, python3 train_voc_images.py --input test_images --output test_images

If you want to test a model with a video, you could run :
- **python3 test_xxx_video.py --input path/to/input/file --output path/to/output/file**: For example, python3 test_coco_video --input test_videos/input.mp4 --output test_videos/output.mp4


