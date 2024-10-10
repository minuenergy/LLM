import os
import json
from datasets import load_dataset
from tqdm import tqdm
import hashlib

# 메인 폴더 및 이미지 폴더 생성
main_folder = "MEDPIX-ClinQA"
image_folder_name = "MEDPIX-ClinQA_images"
image_folder = os.path.join(main_folder, image_folder_name)
os.makedirs(main_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

# 데이터셋 로드
dataset = load_dataset("adishourya/MEDPIX-ClinQA")

# 이미지 저장 함수
def save_image(image, case_id):
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    image_filename = f"{case_id}_{image_hash[:10]}.png"
    image_path = os.path.join(image_folder, image_filename)
    if not os.path.exists(image_path):
        image.save(image_path)
    return os.path.join(image_folder_name, image_filename)

# 데이터셋 처리 함수
def process_dataset(split='train'):
    data = dataset[split]
    new_json_data = []

    print(f"Processing {split} dataset...")
    for item in tqdm(data):
        image_path = save_image(item['image_id'], item['case_id'])

        messages = [
            {"content": f"<image>\n{item['question']}", "role": "user"},
            {"content": item['answer'], "role": "assistant"}
        ]

        new_json_item = {
            "messages": messages,
            "images": [image_path],
            "metadata": {
                "case_id": item['case_id'],
                "mode": item['mode']
            }
        }
        new_json_data.append(new_json_item)

    # JSON 파일 저장
    json_filename = f"MEDPIX-ClinQA_{split}.json"
    json_path = os.path.join(main_folder, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=2)

    return len(new_json_data)

# 메인 실행 부분
if __name__ == "__main__":
    item_count = process_dataset('train')
    print(f"train 처리 완료: {item_count} 항목")
    print(f"\n모든 처리가 완료되었습니다. 데이터는 '{main_folder}' 폴더에 저장되었습니다.")
