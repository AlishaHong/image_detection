import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A


def yolo_to_coco(img_width, img_height, yolo_bbox):
    """
    YOLO 형식의 bounding box를 COCO 형식으로 변환합니다.

    Args:
        img_width (int): 이미지 너비
        img_height (int): 이미지 높이
        yolo_bbox (list): YOLO 형식의 bounding box [x_center, y_center, width, height] (normalized)

    Returns:
        list: COCO 형식의 bounding box [x_min, y_min, width, height] (픽셀 단위)
    """
    x_center, y_center, width, height = yolo_bbox

    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    width = int(width * img_width)
    height = int(height * img_height)

    return [x_min, y_min, width, height]


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, yolo_bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""

    img_width = img.shape[1]
    img_height = img.shape[0]

    bbox = yolo_to_coco(img_width, img_height, yolo_bbox)
    
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img



def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        # class_name = category_id
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


image = cv2.imread('cat_dog/000000386298.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

category_id_to_name = {0: 'cat', 1: 'dog'}


category_ids = []
bboxes = []

f=open('cat_dog/000000386298.txt','r')
while True:
    line = f.readline()
    if not line: break
    ids, xc, yc, w, h= line.split(' ')
    category_ids.append(int(ids))
    bboxes.append([float(xc),float(yc),float(w),float(h)])
    # print(line)
f.close()

print(bboxes)
print(category_ids)

# HorizontalFlip

transform = A.Compose([
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

transform = A.Compose(
    [A.Rotate(limit=(30, 30),p=1,rotate_method='ellipse')],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=1024, min_visibility=0.1)
)


transformed = transform(image=image, bboxes=bboxes, class_labels=category_ids)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']

visualize(transformed_image, transformed_bboxes, transformed_class_labels, category_id_to_name)

print(transformed_bboxes, transformed_class_labels, category_id_to_name)