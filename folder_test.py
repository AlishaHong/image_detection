
import os

# 생성할 폴더 경로
folder_path = os.path.join(os.getcwd(), 'snack_image_data')
print(f"폴더 경로 확인: {folder_path}")

# 폴더가 없으면 생성
if not os.path.exists(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"폴더 생성 성공: {folder_path}")
    except Exception as e:
        print(f"폴더 생성 실패: {e}")
else:
    print(f"폴더가 이미 존재합니다: {folder_path}")