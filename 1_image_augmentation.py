import cv2
import numpy as np
import sys
import os
import glob
import random

# opencv로 bright와 contrast 조절 

class ImageProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = self.get_image_list()
        
    # jpg 원본 리스트 불러오기
    def get_image_list(self):
        data_org = os.path.join(self.data_dir, 'snack_dataOrg2_640')
        file_names = glob.glob(os.path.join(data_org, '*.jpg'))
        return file_names
    
    # 이미지 읽기
    def load_image_by_name(self, img_name):
        for file_name in self.file_names:
            if img_name in file_name:
                print(file_name)
                img = cv2.imread(file_name)
                if img is None:
                    sys.exit('Image load failed')

                return img
        sys.exit(f"{img_name} 파일이 없어요!")


    # 기존 .txt 파일을 읽어오는 함수0
    def load_text_file(self, img_name):
        text_file_path = os.path.join(self.data_dir, 'snack_dataOrg2_640', f'{img_name}.txt')
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"파일 내용:\n{content}")  # 파일 내용을 출력
            return content
        else:
            print(f"{img_name}.txt 파일을 찾을 수 없습니다.")
            return None

    # 새로운 .txt 파일을 저장하는 함수
    def save_text_file(self, new_img_name, content):
        # 텍스트 파일을 저장할 폴더 경로 생성
        new_text_file_dir = os.path.join(os.getcwd(), 'snack_image_data2')
        
        # 경로가 없으면 폴더 생성
        if not os.path.exists(new_text_file_dir):
            os.makedirs(new_text_file_dir)

        # 최종 파일 경로 설정
        file_path = os.path.join(new_text_file_dir, f'{new_img_name}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)  
    
    
    # 저장 경로 생성 함수
    def make_save_path(self,imgName, pre_img_str):
        data_path = os.path.join(os.getcwd(), f'snack_image_data2')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        makeNewFileName = os.path.join(data_path, f'{imgName}_{pre_img_str}.jpg')
        
        return makeNewFileName
    

    # 이미지 저장 함수
    def save_image(self, img, img_name, pre_img_str):
        # make_save_path 함수를 사용하여 이미지 경로 생성
        save_path = self.make_save_path(img_name, pre_img_str)
        
        # 이미지 저장
        cv2.imwrite(save_path, img)


    # 채도와 명도 조절
    def adjust_brightness_and_saturation(self, image, saturation_scale, brightness_scale):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        v = cv2.multiply(v, brightness_scale)
        s = cv2.multiply(s, saturation_scale)
        
        adjusted_hsv_image = cv2.merge([h, s, v])
        return cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)

    # 전처리 과정 실행
    # 원하는 기능만 실행하도록 boolean값 줌 
    def process_image(self, img_name, sb_on = True):
        img = self.load_image_by_name(img_name)
        if img is None:
            print(f"이미지를 로드할 수 없습니다: {img_name}")
            return
        
        # 원본 이미지의 .txt 파일 내용 불러오기
        original_text_content = self.load_text_file(img_name)
        
            
        # 채도, 명도 조절
        if sb_on:
            # print('채도명도')
            saturation_scale = np.arange(0.4, 1.1, 0.1)
            brightness_scale = np.arange(0.4, 1.1, 0.1)       

            for saturation in saturation_scale:
                for brightness in brightness_scale:
                    adjusted_img = self.adjust_brightness_and_saturation(img, saturation, brightness)
                    if adjusted_img is None:
                        print(f"이미지 조정 실패: {img_name} (saturation={saturation}, brightness={brightness})")
                    else:
                        new_img_name = f'{img_name}_{saturation:.1f}_{brightness:.1f}'
                        self.save_image(adjusted_img, new_img_name, 'brightness_saturation')
                        # print(f"{new_img_name} 채도, 명도 조절 이미지 저장 완료")
                    # 새로운 .txt 파일 저장
                    if original_text_content:
                        txt_name = f'{new_img_name}_brightness_saturation'
                        self.save_text_file(txt_name, original_text_content)

    

# 메인 실행 함수
def main():
    data_path = os.getcwd()
    processor = ImageProcessor(data_path)   # 객체 생성
    for file_name in processor.file_names:
        basename = os.path.basename(file_name)
        img_name, _ = os.path.splitext(basename)
        
        # 필요하지 않은 기능은 False로 바꿔서 전처리 기능을 실행하지 않음
        processor.process_image(img_name)
        
        img = processor.load_image_by_name(img_name)  # 인스턴스 메서드로 호출
        height, width, _ = img.shape  # 이미지의 높이와 너비를 얻음
        print(f"Image resolution: {width}x{height}")  # 해상도 출력

if __name__ == "__main__":
    main()