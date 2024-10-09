from datasets import load_dataset
from transformers import AutoTokenizer

def display_token_info(total_tokens, num_items):
    # 총 토큰 수를 billion 단위로 변환
    total_tokens_billion = total_tokens / 1e9

    # 평균 토큰 수 계산
    average_tokens = total_tokens / num_items

    # 결과 출력
    print(f"총 토큰 수: {total_tokens_billion:.6f} billion")
    print(f"데이터셋 항목 수: {num_items}")
    print(f"항목당 평균 토큰 수: {average_tokens:.2f}")

# 데이터셋 로드
dataset = load_dataset("gamino/wiki_medical_terms", split="train")

# Llama 3.1 토크나이저 로드 (실제로는 Llama 2를 사용, Llama 3.1이 공개되면 업데이트 필요)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# 토큰 수를 저장할 변수 초기화
total_tokens = 0

# 데이터셋의 각 항목에 대해 토큰 수 계산
for item in dataset:
    text = item['page_text']
    tokens = tokenizer.encode(text)
    total_tokens += len(tokens)

display_token_info(total_tokens=total_tokens, num_items=len(dataset))
