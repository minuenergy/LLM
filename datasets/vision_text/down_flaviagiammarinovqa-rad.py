from datasets import load_dataset
import os
import json
from PIL import Image
import hashlib

# 데이터셋 로드
ds = load_dataset("flaviagiammarino/vqa-rad")

# 메인 폴더 생성
main_folder = "VQA-RAD"
os.makedirs(main_folder, exist_ok=True)

# 이미지를 저장할 폴더 생성
image_folder_name = "VQA-RAD_images"
image_folder = os.path.join(main_folder, image_folder_name)
os.makedirs(image_folder, exist_ok=True)

def process_dataset(split):
    new_json_data = []

    # 데이터셋의 각 항목에 대해 반복
    for idx, item in enumerate(ds[split]):
        # 이미지 저장
        image = item['image']
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        image_filename = f"{split}_{idx}_{image_hash[:10]}.jpg"
        image_path = os.path.join(image_folder, image_filename)
        image.save(image_path)

        # 메시지 생성
        messages = [
            {"content": f"<image>\n{item['question']}", "role": "user"},
            {"content": item['answer'], "role": "assistant"}
        ]

        # 새로운 JSON 아이템 생성
        new_json_item = {
            "messages": messages,
            "images": [os.path.join(image_folder_name, image_filename)]
        }
        new_json_data.append(new_json_item)

    # JSON 파일로 저장
    json_filename = f"VQA-RAD_{split}.json"
    json_path = os.path.join(main_folder, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=2)

    return len(new_json_data)

# train과 test 데이터셋 처리
train_count = process_dataset('train')
test_count = process_dataset('test')

print(f"이미지가 '{image_folder}' 폴더에 저장되었습니다.")
print(f"train 메타데이터가 'VQA-RAD_train.json' 파일에 저장되었습니다. (항목 수: {train_count})")
print(f"test 메타데이터가 'VQA-RAD_test.json' 파일에 저장되었습니다. (항목 수: {test_count})")
