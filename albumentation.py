import os
import cv2
from matplotlib import pyplot as plt
import albumentations as A

def yolo_to_coco(img_width, img_height, yolo_bbox):
    """
    YOLO 형식의 bounding box를 COCO 형식으로 변환합니다.
    """
    x_center, y_center, width, height = yolo_bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    width = int(width * img_width)
    height = int(height * img_height)
    return [x_min, y_min, width, height]


def visualize_bbox(img, yolo_bbox, class_name, color=(255, 0, 0), thickness=2):
    # 단일 바운딩 박스 시각화
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
    """
    이미지에 여러 개의 bounding box를 시각화합니다.
    """
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def read_bounding_boxes(file_path):
    # txt 파일에서 바운딩박스와 클래스번호 읽어온 뒤 리스트에 담아주기
    category_ids = []
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            ids, xc, yc, w, h = line.split(' ')
            category_ids.append(int(ids))
            bboxes.append([float(xc), float(yc), float(w), float(h)])
    return bboxes, category_ids


def apply_transformations(image, bboxes, category_ids, rotation_angle, flip, brightness_limit, contrast_limit):
    # 여러개의 전처리값을 이미지와 바운딩 박스에 적용하기
    
    transforms = []
    transform_description = f"rotate_{rotation_angle}"

    # 회전 변환 추가
    transforms.append(A.Rotate(limit=(rotation_angle, rotation_angle), p=1, rotate_method='ellipse'))

    # 플립 변환 추가
    if flip:
        transforms.append(A.HorizontalFlip(p=1))
        transform_description += "_flip"
    
    

    # 밝기/대비 변환 추가
    transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1))
    transform_description += f"_brightness_{brightness_limit}_contrast_{contrast_limit}"

    transform = A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=1024, min_visibility=0.1))
    transformed = transform(image=image, bboxes=bboxes, class_labels=category_ids)
    return transformed['image'], transformed['bboxes'], transformed['class_labels'], transform_description


def process_images_in_folder(folder_path, output_folder):
    # 지정한 폴더의 모든 파일을 읽어옴
    for filename in os.listdir(folder_path):
        # jpg파일경로 가져오기
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            # txt파일경로 가져오기
            txt_path = image_path.replace('.jpg', '.txt')
            
            if not os.path.exists(txt_path):
                continue  # JPG 파일에 해당하는 TXT 파일이 없으면 넘어감

            # 이미지와 bounding box 불러오기
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, category_ids = read_bounding_boxes(txt_path)

            # 다중 변환 적용 (10도에서 180도까지 회전, 플립, 밝기/대비 조합)
            # for rotation_angle in range(10, 361, 30): 
            #     for flip in [True, False]:  # 플립 적용/미적용
            #         for brightness_limit in [0.1, 0.2, 0.3]:  # 밝기 범위 설정
            #             for contrast_limit in [0.1, 0.2, 0.3]:  # 대비 범위 설정
                            # transformed_image, transformed_bboxes, transformed_class_labels, applied_transform = apply_transformations(
                            #     image, bboxes, category_ids, rotation_angle, flip, brightness_limit, contrast_limit
                            # )

            transformed_image, transformed_bboxes, transformed_class_labels, applied_transform = apply_transformations(
            image, bboxes, category_ids, rotation_angle, flip, brightness_limit, contrast_limit)
            
            # 새롭게 생성된 파일명 설정 (원본 파일명 + 변환 정보)
            base_filename = os.path.splitext(filename)[0]
            new_image_name = f"{base_filename}_{applied_transform}.jpg"
            new_txt_name = f"{base_filename}_{applied_transform}.txt"

            # 결과 저장
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)  # OpenCV 형식으로 변환
            cv2.imwrite(os.path.join(output_folder, new_image_name), transformed_image)

            with open(os.path.join(output_folder, new_txt_name), 'w') as f:
                for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                    bbox_str = ' '.join(map(str, bbox))
                    f.write(f"{class_id} {bbox_str}\n")

            print(f"파일 저장 완료: {new_image_name}, {new_txt_name}")


# 예제 사용
input_folder = 'input_images'  # 원본 이미지와 TXT 파일이 있는 폴더
output_folder = 'output_images'  # 변환된 이미지와 TXT 파일을 저장할 폴더

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_folder(input_folder, output_folder)
print(f'변환된 파일 개수: {len({output_folder})}')

