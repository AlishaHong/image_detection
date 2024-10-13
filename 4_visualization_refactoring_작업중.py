import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 학습된 모델 가져오기
model = YOLO("runs/detect/1st_open_reo_8_n_T4/weights/best.pt")

# 클래스 이름 리스트
class_name = ['snack', 'hershey', 'eclipse']

# 테스트 파일 폴더 경로
folder_path = 'test_snack_detect'

# 폴더에 있는 모든 이미지 파일 가져오기
files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 파일 예측 및 시각화
for file in files:
    filepath = os.path.join(folder_path, file)
    org_image = cv2.imread(filepath)
    resized_image = cv2.resize(org_image, (860, 600))  # 이미지 크기 조정
    height, width, _ = resized_image.shape
    # 모델 예측
    results = model(resized_image)

    # 각 이미지의 예측 결과 순회하며 바운딩 박스와 텍스트 추가
    for result in results:
        for box in result.boxes:
            # 바운딩 박스 좌표 추출 및 보정
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            
            # 경계값 보정 
            # 최소값과 최대값이 일정값을 넘어가지 않도록 하기 위함
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(width, int(x_max)), min(height, int(y_max))

            # 레이블과 점수 추출
            confidence = box.conf.item()
            label = int(box.cls.item())
            label_name = class_name[label]

            # 바운딩 박스 그리기 (OpenCV)
            # 기존의 코드는 matplotlib으로 사각형을 그려서 
            # width,height값이 필요했지만 opencv함수는 두개의 좌표값만 있으면 사각형을 그릴 수 있음
            cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # 텍스트 추가 (OpenCV)
            text = f'{label_name}: {confidence:.2f}'
            cv2.putText(resized_image, text, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # OpenCV로 처리된 이미지를 Matplotlib으로 시각화
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
    plt.axis('off')
    plt.title(f'{file}')
    plt.show()
