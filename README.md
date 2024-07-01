make caption with llava

tested with coco dataset. 

using coco's label + image -> llava -> caption

I'm trying to more accurate and detatiled caption with detection label

python main.py --model-path llava_weights/llava-v1.5-7b/ --image-dir /workspace/mount/SSD_4T_a/minwoo/data/detect/coco/train2017 \n
--ann-file /workspace/mount/SSD_4T_a/minwoo/data/detect/coco/annotations/instances_minitrain2017.json \n
--output-file instances_minitrain2017_llavacap.json --query "Answer with provided hint below."
