import os
import json
import hashlib
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 데이터셋 로드
ds = load_dataset("flaviagiammarino/path-vqa")

# 메인 폴더 및 이미지 폴더 생성
main_folder = "PathVQA"
image_folder_name = "PathVQA_images"
image_folder = os.path.join(main_folder, image_folder_name)
os.makedirs(main_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

# 이미지 저장 함수
def save_image(image, image_id):
    image_path = os.path.join(image_folder, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        image.save(image_path)
    return image_id

def process_dataset(split):
    data = ds[split]
    new_json_data = []
    image_ids = set()

    print(f"{split} 데이터 처리 중...")
    for idx, item in enumerate(tqdm(data)):
        # 이미지 데이터로부터 해시 생성
        image_hash = hashlib.md5(item['image'].tobytes()).hexdigest()
        image_id = f"{split}_{idx}_{image_hash[:10]}"
        image_ids.add(image_id)

        messages = [
            {"content": f"<image>\n{item['question']}", "role": "user"},
            {"content": item['answer'], "role": "assistant"}
        ]

        new_json_item = {
            "messages": messages,
            "images": [os.path.join(image_folder_name, f"{image_id}.jpg")]
        }
        new_json_data.append(new_json_item)

    # 이미지 저장
    print(f"{split} 이미지 저장 중...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(save_image, item['image'], f"{split}_{idx}_{hashlib.md5(item['image'].tobytes()).hexdigest()[:10]}"): idx for idx, item in enumerate(data)}
        for future in tqdm(as_completed(future_to_id), total=len(data)):
            idx = future_to_id[future]
            try:
                future.result()
            except Exception as exc:
                print(f'{split} 데이터의 {idx}번째 이미지 저장 중 오류 발생: {exc}')

    # JSON 파일로 저장
    json_filename = f"PathVQA_{split}.json"
    json_path = os.path.join(main_folder, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=2)

    return len(new_json_data), len(image_ids)

# 각 데이터셋 처리
for split in ['train', 'validation', 'test']:
    try:
        item_count, image_count = process_dataset(split)
        print(f"{split} 처리 완료:")
        print(f"- 데이터가 'PathVQA_{split}.json' 파일에 저장되었습니다. (항목 수: {item_count})")
        print(f"- 이미지 수: {image_count}")
    except Exception as e:
        print(f"{split} 데이터셋 처리 중 오류 발생: {e}")

print(f"\n모든 이미지가 '{image_folder}' 폴더에 저장되었습니다.")
