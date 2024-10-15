import os
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import random

def yolo_to_coco(img_width, img_height, yolo_bbox):
    x_center, y_center, width, height = yolo_bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    width = int(width * img_width)
    height = int(height * img_height)
    return [x_min, y_min, width, height]

def read_bounding_boxes(file_path):
    category_ids, bboxes = [], []
    with open(file_path, 'r') as f:
        for line in f:
            ids, xc, yc, w, h = line.split()
            category_ids.append(int(ids))
            bboxes.append([float(xc), float(yc), float(w), float(h)])
    return bboxes, category_ids

def apply_random_transformations(image, bboxes, category_ids):
    rotate_angle = random.randint(-90, 90)
    gauss_var_limit = random.uniform(1000.0, 4000.0)

    transforms = A.Compose([
        A.RandomCrop(width=600, height=600, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(rotate_angle, rotate_angle), p=0.8, rotate_method='ellipse'),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(gauss_var_limit, gauss_var_limit), p=0.3),
        A.MotionBlur(blur_limit=9, p=0.7)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    transformed = transforms(image=image, bboxes=bboxes, class_labels=category_ids)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

def process_images_and_labels(input_images_folder, input_labels_folder, output_folder, num_variations=400):
    for root, _, files in os.walk(input_images_folder):
        for filename in files:
            if filename.endswith('.jpg'):
                image_path = os.path.join(root, filename)
                label_path = image_path.replace(input_images_folder, input_labels_folder).replace('.jpg', '.txt')

                if not os.path.exists(label_path):
                    continue

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bboxes, category_ids = read_bounding_boxes(label_path)

                output_subfolder = root.replace(input_images_folder, output_folder)
                label_output_subfolder = output_subfolder.replace('images', 'labels')

                os.makedirs(output_subfolder, exist_ok=True)
                os.makedirs(label_output_subfolder, exist_ok=True)

                for i in range(num_variations):
                    transformed_image, transformed_bboxes, transformed_class_labels = apply_random_transformations(
                        image, bboxes, category_ids)

                    base_filename = os.path.splitext(filename)[0]
                    new_image_name = f"{base_filename}_{i+1}.jpg"
                    new_label_name = f"{base_filename}_{i+1}.txt"

                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_subfolder, new_image_name), transformed_image)

                    with open(os.path.join(label_output_subfolder, new_label_name), 'w') as f:
                        for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                            bbox_str = ' '.join(map(str, bbox))
                            f.write(f"{class_id} {bbox_str}\n")

                    print(f"파일 저장 완료: {new_image_name}, {new_label_name}")

input_images_folder = 'origin_albu_test/train/images'
input_labels_folder = 'origin_albu_test/train/labels'
output_folder = 'origin_albu_last_data'

os.makedirs(output_folder, exist_ok=True)
process_images_and_labels(input_images_folder, input_labels_folder, output_folder, num_variations=44)
