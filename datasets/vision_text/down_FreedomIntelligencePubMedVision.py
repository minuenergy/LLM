from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# 메인 폴더 및 이미지 폴더 생성
main_folder = "PubMedVision"
image_folder_name = "PubMedVision_images"
image_folder = os.path.join(main_folder, image_folder_name)
os.makedirs(main_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

# 파일 다운로드 함수
def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

# ZIP 파일 추출 함수
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# 데이터셋 처리 함수
def process_dataset(subset_name):
    print(f"Processing {subset_name}...")

    # JSON 파일 다운로드
    json_url = f"https://huggingface.co/datasets/FreedomIntelligence/PubMedVision/resolve/main/PubMedVision_{subset_name}_VQA.json"
    json_path = os.path.join(main_folder, f"{subset_name}_VQA.json")
    download_file(json_url, json_path)

    # JSON 파일 로드 및 처리
    with open(json_path, 'r') as f:
        data = json.load(f)

    new_json_data = []
    for item in tqdm(data, desc="Processing items"):
        messages = [
            {"content": f"<image>\n{item['conversations'][0]['value']}", "role": "user"},
            {"content": item['conversations'][1]['value'], "role": "assistant"}
        ]

        # 이미지 경로 수정
        images = [os.path.join(image_folder_name, img) for img in item['image']]

        new_json_item = {
            "messages": messages,
            "images": images,
            "metadata": {
                "id": item['id'],
                "body_part": item['body_part'],
                "modality": item['modality']
            }
        }
        new_json_data.append(new_json_item)

    # 처리된 JSON 저장
    processed_json_path = os.path.join(main_folder, f"PubMedVision_{subset_name}.json")
    with open(processed_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=2)

    return len(new_json_data)

# 이미지 ZIP 파일 다운로드 및 추출
def download_and_extract_images():
    base_url = "https://huggingface.co/datasets/FreedomIntelligence/PubMedVision/resolve/main/"
    zip_files = [f"images_{i}.zip" for i in range(20)]  # 0부터 19까지

    for zip_file in zip_files:
        zip_url = base_url + zip_file
        zip_path = os.path.join(main_folder, zip_file)
        print(f"Downloading {zip_file}...")
        download_file(zip_url, zip_path)

        print(f"Extracting {zip_file}...")
        extract_zip(zip_path, image_folder)

        # 추출 후 ZIP 파일 삭제
        os.remove(zip_path)

# 메인 실행 부분
if __name__ == "__main__":
    # 이미지 다운로드 및 추출
    download_and_extract_images()

    # 각 서브셋 처리
    subsets = ["Alignment", "InstructionTuning"]
    for subset in subsets:
        item_count = process_dataset(subset)
        print(f"{subset} 처리 완료: {item_count} 항목")

    print(f"\n모든 처리가 완료되었습니다. 데이터는 '{main_folder}' 폴더에 저장되었습니다.")
