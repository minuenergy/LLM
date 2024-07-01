import argparse
import torch
import numpy as np
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from tqdm import tqdm
from coco_91_to_80_cls import coco91_to_coco80_class, coco80_class
import os
from PIL import Image, ImageDraw, ImageFont



class ImageBatchProcessor:
    def __init__(self, class_names, output_dir='examples'):
        self.class_names = class_names
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.img_count = 0

    def process_batch(self, images, bboxes, labels):
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]


        num_cols = min(batch_size, 4)  # 최대 4열로 설정
        num_rows = (batch_size - 1) // num_cols + 1  # 행 수 계산

        total_width = width * num_cols
        total_height = height * num_rows
        combined_image = Image.new('RGB', (total_width, total_height))

        font_size = min(width, height) // 20  # 이미지 크기에 따라 폰트 크기 조정
        
        print("labels:", len(labels))
        print("bboxes:", len(bboxes))
        for j in range(batch_size):
            img = images[j].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min()) * 255  # 정규화 해제 및 스케일링
            img = img.astype(np.uint8)
            pil_img = Image.fromarray(img)

            col_idx = j % num_cols
            row_idx = j // num_cols
            combined_image.paste(pil_img, (col_idx * width, row_idx * height))
            draw = ImageDraw.Draw(combined_image)

            for (x1, y1, x2, y2), label in zip(bboxes[j], labels[j]):
                draw.rectangle([((col_idx * width) + x1, (row_idx * height) + y1),
                                ((col_idx * width) + x2, (row_idx * height) + y2)],
                                outline='red', width=2)
                text = label
                font = ImageFont.truetype("arial.ttf", font_size)
                text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), text, font=font)
                text_width = text_right - text_left
                text_height = text_bottom - text_top
                text_x = (col_idx * width) + x1 + (x2 - x1 - text_width) // 2
                text_y = (row_idx * height) + y1 - text_height - 1

                draw.text((text_x, text_y), text, fill='red', font=font)

        return combined_image

    def save_combined_image(self, combined_image, filename='combined_batch_image'):
        self.img_count+=1
        filename = f'{filename}_{str(self.img_count)}.jpg'
        combined_image.save(os.path.join(self.output_dir, filename))

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def load_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    return model_name, tokenizer, model, image_processor, context_len 

from torch.nn.utils.rnn import pad_sequence



def tokenizer_image_token_batch(prompts, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    batch_input_ids = []
    for prompt in prompts:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        batch_input_ids.append(input_ids)
    batch_input_ids_padded = pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in batch_input_ids], batch_first=True)    
    
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(batch_input_ids_padded, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    
    return batch_input_ids



def question_answer(args, data_loader, model_name, tokenizer, model, image_processor, context_len):
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode




    results = []
    
    class_names = coco80_class()
    processor = ImageBatchProcessor(class_names)

    i=0
    model.eval()
    with torch.no_grad():
        for images, img_ids, in_context, bboxes, cls   in tqdm(data_loader):
            i+=1
            images = images.to('cuda')
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            in_context_prompt = [prompt.split("<image>")[0]+cont+" <image>" for cont in in_context]
            
          
            if i <=5:
                combined_image = processor.process_batch(images, bboxes, cls)
                processor.save_combined_image(combined_image)

            '''
            input_ids = (
                tokenizer_image_token_batch(in_context_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=images.size,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            results.append(output)
            print("output:", output)
            '''
            ## batch.. but padding is not good option.. i have to change. . hmm..
            results={}
            for img, img_id, prom in zip(images, img_ids, in_context_prompt):
                print("#"*100)
                print("img_id:", img_id)
                print("prompt:", prom +'\n')
                input_ids = (
                    tokenizer_image_token(prom, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images,
                        image_sizes=images.size,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                results["img_id"] = output
                print("output:", output)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
