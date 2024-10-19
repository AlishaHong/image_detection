from ultralytics import YOLO
import cv2
import logging
import os
import time

# 로깅 수준을 WARNING으로 설정하여 불필요한 출력 억제
logging.getLogger().setLevel(logging.WARNING)

model_path = 'best_pt_정리/0_리사이즈여부_1st_open_reo_8_n_1660/best.pt'
# YOLO11 나노 모델 로드
model = YOLO(model_path, verbose=False)  # 모델 파일 경로를 사용하여 로드

# 동영상 파일 경로 (미리 촬영된 영상)
video_path = 'snack_video/recorded_video_2.avi'  # 여기에 동영상 파일 경로 입력
video = cv2.VideoCapture(video_path)  # 동영상 파일을 사용

# 동영상 FPS 가져오기
fps = video.get(cv2.CAP_PROP_FPS)
frame_time = int(1000 / fps)  # 프레임 간 시간 (밀리초 단위)

# 녹화 관련 변수
recording = False
out = None

# 저장된 모델명으로 폴더이름을 추출
def make_folder_name(model):
    folder_name = model.split('/')[1] + 'result_video'
    return folder_name

# 영상에 번호를 부여하여 덮어씌워지지 않게 함
def get_unique_filename(folder_name , base_name='recorded_video', ext='.avi'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # 폴더가 없으면 생성
    i = 1
    filename = os.path.join(folder_name, f"{base_name}{ext}")   # ext : 파일 확장자
    while os.path.exists(filename):
        filename = os.path.join(folder_name, f"{base_name}_{i}{ext}")
        i += 1
    return filename

folder_name = make_folder_name(model_path)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # YOLO 모델을 사용하여 객체 감지
    results = model(frame)

    # 감지된 객체들을 프레임에 표시
    annotated_frame = results[0].plot()

    # 만약 녹화 중이면, 프레임을 파일에 저장
    if recording and out is not None:
        out.write(annotated_frame)

    cv2.imshow('Real-Time Object Detection', annotated_frame)
    key = cv2.waitKey(frame_time)  # 동영상의 원래 FPS에 맞춰 대기 시간 설정

    if key == ord('r'):  # 'r' 키를 누르면 녹화 시작
        if not recording:
            recording = True
            # 고유한 파일 이름을 생성
            video_filename = get_unique_filename(folder_name)
            # 비디오 파일 저장 설정 (코덱, 파일 이름, FPS, 프레임 크기)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, fps, (annotated_frame.shape[1], annotated_frame.shape[0]))
            print(f"녹화 시작: {video_filename}")
    
    elif key == ord('s'):  # 's' 키를 누르면 녹화 종료
        if recording:
            recording = False
            out.release()
            out = None
            print("녹화 종료")

    elif key == 27:  # ESC 키를 누르면 종료
        break

# 리소스 정리
video.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
