import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from pycocotools.coco import COCO
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_llava import load_model, question_answer

from llava.mm_utils import expand2square, process_anyres_image 
import numpy as np
from coco_91_to_80_cls import coco91_to_coco80_class, coco80_class

def pad_collate_fn(batch):
    images, img_ids, in_context, bbox, coco_cls = zip(*batch)

    # Find max height and width
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    # Pad images
    # padded_images = []
    # for image in images:
    #     #padded_image = torch.nn.functional.pad(image, (0, max_width - image.shape[2], 0, max_height - image.shape[1]))
    #     #padded_images.append(padded_image)
    padded_images = images
    # Stack images
    padded_images = torch.stack(padded_images)

    # Bboxes와 classes를 리스트로 유지
    return padded_images, img_ids, list(in_context), list(bbox), list(coco_cls)

def process_images(image, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    if image_aspect_ratio == 'pad':
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    elif image_aspect_ratio == "anyres":
        image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
    else:
        return image_processor(image, return_tensors='pt')['pixel_values']

    return image

class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, image_processor, model_cfg, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        #self.ids = list(self.coco.imgs.keys())
        self.ids = self.coco.getImgIds()
        self.transform = transform

        ## llava set 
        self.image_processor = image_processor
        self.model_cfg = model_cfg

        self.convert_91to80 = coco91_to_coco80_class()
        self.class_names = coco80_class()

        ## prompt
        self.instruct_prompt='There can be multiple classes in an image, and provide exist object and stuff class. The contents of {} are the classes present in the image. Describe the image with only provided class.'
        
        #'There can be multiple classes in an image, and each class is represented in the form [x, y, w, h at class]. The contents of {} are the classes present in the image. please generate a caption that focuses on the provided classes'
    def apply_transform(self, image, annotations):
        # 이미지 크기를 먼저 저장
        original_size = image.size
        
        # transform 적용
        if self.transform:
            image, annotations = self.apply_transform(image, annotations)

        # 리사이즈된 이미지 크기를 얻음
        resized_size = image.size(1), image.size(2)  # (height, width)
        
        # 원본과 리사이즈된 크기를 기준으로 bbox 조정
        scale_x = resized_size[1] / original_size[0]
        scale_y = resized_size[0] / original_size[1]

        for ann in annotations:
            bbox = ann['bbox']
            bbox[0] *= scale_x
            bbox[1] *= scale_y
            bbox[2] *= scale_x
            bbox[3] *= scale_y
            ann['bbox'] = bbox

        return image, annotations 

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.ids[image_index], iscrowd=False)

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)

        boxes = []
        coco_classes = []
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            cls_n = self.convert_91to80[a['category_id']]
            if cls_n is None:
                continue
            
            cls_n = self.class_names[cls_n-1]
            boxes.append(a['bbox'])
            coco_classes.append(cls_n)
        
        return boxes, coco_classes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(path).convert('RGB')

        # Bbox 정보 로딩
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annotations = self.coco.loadAnns(ann_ids)

        image = process_images(image, self.image_processor, self.model_cfg) 

        annot, coco_classes = self.load_annotations(idx)
        
        formatted_annotations = ''

        # 각 annotation에 대해 처리
        for bbox, cls in zip(annot, coco_classes):
            if cls is not None:
                for_ann = str(bbox)+' at '+str(cls)+', '
                #for_ann = str(cls)+', ' 
                formatted_annotations += for_ann
            else:
                continue
            
        formatted_annotations = "{" + formatted_annotations + "}" + self.instruct_prompt

        return image.half(), img_id, formatted_annotations, annot, coco_classes


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


def eval_model(args):
    # 모델과 토크나이저 로드
    model_name, tokenizer, model, image_processor, context_len = load_model(args)
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        #transforms.ToTensor(),
    ])
    
    dataset = COCODataset(img_dir=args.image_dir, ann_file=args.ann_file, image_processor=image_processor, model_cfg=model.config, transform=transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,  collate_fn=pad_collate_fn) #collate_fn=lambda x: x)
    outputs = question_answer(args, data_loader, model_name, tokenizer, model, image_processor, context_len)

    return outputs

import json
from pycocotools.coco import COCO

def save_result(args, results):
    # Load your existing COCO dataset
    coco = COCO(args.ann_file)

    # Iterate over each image ID in the COCO dataset
    for img_id in coco.imgs.keys():
        if img_id in results:
            # Update the COCO dataset with 'caption' field
            coco.imgs[img_id]['caption'] = results[img_id]
        else:
            # Handle cases where LLM did not produce an output for some images
            coco.imgs[img_id]['caption'] = "XX"  # Or any default value you prefer

    # Save the updated COCO dataset to a new JSON file
    new_json_file = args.output_file
    with open(new_json_file, 'w') as f:
        json.dump(coco.dataset, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--ann-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    
    #parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")

    
    args = parser.parse_args()

    results = eval_model(args)

    save_result(args.ann_file, results, args.output_file)
