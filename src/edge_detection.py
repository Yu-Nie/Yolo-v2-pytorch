import cv2
import numpy as np
import xml.etree.ElementTree as ET


def get_ratio(image_path, xmin, xmax, ymin, ymax):
    # image_path = 'data/VOCdevkit/VOC2012/SegmentationObject/2007_002597.png'
    image = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    edge = cv2.Canny(blurred, 10, 70)
    segment = edge[ymin:ymax, xmin:xmax]
    edge_points = []
    for y in range(0, segment.shape[0]):
        for x in range(0, segment.shape[1]):
            if segment[y, x] != 0:
                edge_points = edge_points + [[x, y]]
    edge_points = np.array(edge_points)
    total = len(edge_points)
    if total == 0:
        print(image_path, "can't find edge!")
        return

    xlen, ylen = segment.shape
    xdiff = int(xlen / 4)
    ydiff = int(ylen / 4)
    ratio = []
    for i in range(4):
        y1 = ymin + ydiff * i
        for j in range(4):
            x1 = xmin + xdiff * j
            grid = edge[y1:(y1 + ydiff), x1:(x1 + xdiff)]
            grid_edge_points = []
            for y in range(0, grid.shape[0]):
                for x in range(0, grid.shape[1]):
                    if grid[y, x] != 0:
                        grid_edge_points = grid_edge_points + [[x, y]]
            grid_edge_points = np.array(grid_edge_points)
            ratio.append(len(grid_edge_points) / total)
    # print(ratio)
    return ratio


def pretty_xml(element, indent, newline, level=0):
    if element:
        if (element.text is None) or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # if uncomment the else, the text of Element will also be a newline
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # if not last element of list, keep same indent
            subelement.tail = newline + indent * (level + 1)
        else:  # if last element of list, reduce indent by 1
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # recursively change sub-element

