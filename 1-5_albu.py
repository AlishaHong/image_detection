import os
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import random
import numpy as np

def yolo_to_coco(img_width, img_height, yolo_bbox):
    x_center, y_center, width, height = yolo_bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    width = int(width * img_width)
    height = int(height * img_height)
    return [x_min, y_min, width, height]

def visualize_bbox(img, yolo_bbox, class_name, color=(255, 0, 0), thickness=2):
    img_width = img.shape[1]
    img_height = img.shape[0]
    bbox = yolo_to_coco(img_width, img_height, yolo_bbox)
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def read_bounding_boxes(file_path):
    category_ids = []
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.split(' ')
            # 첫 번째 값은 클래스 ID, 그 다음 4개의 값은 바운딩 박스 좌표
            class_id = int(data[0])  # 첫 번째 값은 클래스 ID
            bbox = [float(data[1]), float(data[2]), float(data[3]), float(data[4])]  # 그 다음 4개의 값이 바운딩 박스
            category_ids.append(class_id)
            bboxes.append(bbox)
    return bboxes, category_ids

def apply_random_transformations(image, bboxes, category_ids):
    transform_description = []

    rotate_angle = random.randint(-44, 44)
    gauss_var_limit = random.uniform(1000.0, 4000.0)

    transforms = A.Compose([
        A.RandomCrop(width=600, height=600, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(rotate_angle, rotate_angle), p=0.8, rotate_method='ellipse'),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(gauss_var_limit, gauss_var_limit), p=0.3),
        A.MotionBlur(blur_limit=9, p=0.7)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2,clip=True))

    transformed = transforms(image=image, bboxes=bboxes, class_labels=category_ids)
    # print("여기까지 오니?")
    # 변환 후 바운딩 박스 좌표를 [0, 1] 범위로 클리핑
    transformed_bboxes = []
    transformed_class_labels = []

    for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
        # 바운딩 박스 좌표를 [0, 1] 범위로 강제 클리핑
        bbox = np.clip(np.array(bbox), 0, 1)

        # 만약 bboxes의 모든 값이 음수인 경우 무시
        if all(coord < 0 for coord in bbox):
            print(f"무시된 바운딩 박스 (모든 값이 음수): {bbox}")
            continue  # 유효하지 않은 바운딩 박스는 무시하고 다음으로 넘어감

        transformed_bboxes.append(bbox.tolist())  # 다시 리스트로 변환하여 저장
        transformed_class_labels.append(class_id)



    return transformed['image'], transformed_bboxes, transformed_class_labels, "_".join(transform_description)

def process_images_in_folder_random(image_folder, label_folder, num_variations=400):
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            txt_path = os.path.join(label_folder, filename.replace('.jpg', '.txt'))

            if not os.path.exists(txt_path):
                continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, category_ids = read_bounding_boxes(txt_path)

            for i in range(num_variations):
                transformed_image, transformed_bboxes, transformed_class_labels, applied_transform = apply_random_transformations(
                    image, bboxes, category_ids
                )

                if not applied_transform:
                    applied_transform = f"{i+1}"

                base_filename = os.path.splitext(filename)[0]
                new_image_name = f"{base_filename}_{applied_transform}.jpg"
                new_txt_name = f"{base_filename}_{applied_transform}.txt"

                if transformed_bboxes:
                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(image_folder, new_image_name), transformed_image)

                    with open(os.path.join(label_folder, new_txt_name), 'w') as f:
                        for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                            class_id = int(class_id)
                            bbox_str = ' '.join(map(str, bbox))
                            f.write(f"{class_id} {bbox_str}\n")

                    print(f"파일 저장 완료: {new_image_name}, {new_txt_name}\n")

def load_and_visualize(image_path, label_path, category_id_to_name):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, category_ids = read_bounding_boxes(label_path)
    visualize(image, bboxes, category_ids, category_id_to_name)

# 예제 사용
image_folder = 'original_x5_splited/train/images'
label_folder = 'original_x5_splited/train/labels'

process_images_in_folder_random(image_folder, label_folder, num_variations=4)
