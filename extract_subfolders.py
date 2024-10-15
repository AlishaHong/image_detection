import os
import zipfile

def extract_zip(file_path, extract_to):
    # zip파일 압축해제
    with zipfile.ZipFile(file_path, 'r') as zip:
        zip.extractall(extract_to)
    print(f"압축해제: {file_path}")
    
    # 압축 해제 완료 후 ZIP 파일 삭제
    os.remove(file_path)
    print(f"zip파일삭제: {file_path}")

def extract_all_zip_in_folder(folder_path):
    # 모든 하위폴더의 압축을 해제
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                extract_to = os.path.splitext(file_path)[0]  # 파일명에 해당하는 폴더 생성
                if not os.path.exists(extract_to):
                    os.makedirs(extract_to)
                extract_zip(file_path, extract_to)

if __name__ == "__main__":
    folder_path = '[원천]양추정_VAL'
    extract_all_zip_in_folder(folder_path)