import requests
from bs4 import BeautifulSoup
import os

# 이미지 크롤링 함수
def crawl_images(query, folder_path, num_images=300):
    base_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0"}
    image_count = 1  # 이미지 번호를 1로 시작

    for page in range(1, num_images//10 + 1):  # 각 페이지당 약 10개의 이미지가 있다고 가정
        if image_count > num_images:
            break  # 필요한 이미지 수에 도달하면 중지

        # Google 검색 페이지 URL 구성
        url = f"{base_url}&start={page*10}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # 이미지 URL 추출
        images = soup.find_all("img")

        if not images:
            print(f"이미지 없음. 페이지 {page}에서 중지.")
            break  # 이미지가 더 이상 없으면 중지

        # 이미지를 폴더에 저장
        for img in images:
            img_url = img.get('src')
            if img_url and image_count <= num_images:  # 유효한 이미지 URL인지 확인 및 이미지 수 제한
                try:
                    img_data = requests.get(img_url).content
                    image_filename = os.path.join(folder_path, f"{image_count}.jpg")  # 이미지 이름을 번호로 설정
                    with open(image_filename, "wb") as f:
                        f.write(img_data)
                    print(f"{query} - 이미지 {image_count} 저장 완료.")
                    image_count += 1  # 이미지 번호 증가
                except:
                    print(f"{query} - 이미지 {image_count} 저장 실패.")

# 폴더 생성 및 크롤링 실행
def create_folders_and_crawl(names, base_directory):
    for idx, name in enumerate(names, start=19):
        folder_name = f"{idx:02d}"  # 폴더 이름을 01, 02, ..., 15로 설정
        folder_path = os.path.join(base_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"폴더 생성 완료: {folder_path}")
        crawl_images(name, folder_path)

# 이름 리스트
names = [
    "강하늘", "조정석", "프로미스나인-백지헌", "배우 공유", "변요한", "하정우", "이종석"
]

# 기본 저장 경로
base_directory = "./people"

# 폴더 생성 및 이미지 크롤링 시작
create_folders_and_crawl(names, base_directory)