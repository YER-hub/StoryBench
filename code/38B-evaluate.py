import numpy as np
import torch
import torchvision.transforms as T
import math
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision.transforms.functional   import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
def split_model(model_name):
    device_map = {}
    gpu_list=[0,1, 2, 3,4] # List of GPU IDs to use, e.g., [0, 1, 2, 3, 4 ]
    world_size = len(gpu_list)
    #torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = gpu_list[i]
            layer_cnt += 1
    device_map['vision_model'] = gpu_list[0]  
    device_map['mlp1'] = gpu_list[0]   
    device_map['language_model.model.tok_embeddings'] = gpu_list[0]  
    device_map['language_model.model.embed_tokens'] = gpu_list[0]  
    device_map['language_model.output'] = gpu_list[1] 
    device_map['language_model.model.norm'] = gpu_list[1]  
    device_map['language_model.lm_head'] = gpu_list[1]  
    device_map[f'language_model.model.layers.{num_layers - 1}'] = gpu_list[1]  
    device_map['language_model.model.rotary_emb'] = gpu_list[1]  

    return device_map

path = "/data/yekaiyang/bench-pipline/InternVL2_5-38B" # Path to the model directory
device_map = split_model('InternVL2_5-38B')
print(device_map)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False,pad_token='<pad>')

#The above is the model-loading code
#The following is the model-inference code

import json
import torch
import pandas as pd
import os

device = torch.device('cuda:0')

generation_config = dict(max_new_tokens=1024, do_sample=True)
pixel_values = None  

json_files = ['Zero-hop.json','One-hop.json','Two-hop.json','Three-hop.json'] # such as Zero-hop.json, One-hop.json, etc.
image_folders = ['SD3','SDXL','Flux'] # such as SD3-One-hop 
json_base_path = r'./data/multi-hop-data/'  # Path to the DATA JSON files
Image_base_path = r'./Image/'  # Path to the image folders

all_results = []

for json_file in json_files:
    json_path = os.path.join(json_base_path, json_file)
    with open(json_path, 'r') as file:
        prompts_data = json.load(file)

    for folder in image_folders:
        results = []
        image_folder_path = os.path.join(Image_base_path, f"{folder}-{json_file.split('.')[0]}")
        output_csv = f"38B-{folder}-{json_file.split('.')[0]}.csv"

        for category, prompt_list in prompts_data.items():
            for idx, prompt_data in enumerate(prompt_list):
                image_name = f"{category}_{idx + 1}.png"
                image_path = os.path.join(image_folder_path, image_name)

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(device)

                prompt = prompt_data['prompt']
                questions = prompt_data['questions']

                print(f"Processing image: {image_name} in {folder}")
                answers = {}

                # Iterate through the questions and process each one hierarchically
                for level, level_questions in questions.items():
                    # Skip levels other than "Level 1"
                    if level != "Level 1":
                        continue
                    
                    for q_idx, question in enumerate(level_questions):
                        instruction = (
                                "Now,please score based on how well the content of the image matches the description in the question. The scoring criteria are as follows:\n "
                                "2 point for fully matching (the image content completely aligns with the question description, with no ambiguity or deviation),\n "
                                "1 point for partially matching (the image includes elements related to the question, but with some minor discrepancies in details or aspects), \n"
                                "0 point for not matching at all (the image does not show what is described in the question, or is completely inconsistent). \n"
                                "Please provide only the score, without any additional explanation.for example:0,1,2"
                            )
                        full_question = f"{question} {instruction}"
                        print(f"{level}_{q_idx + 1}: {full_question}")

                        response = model.chat(tokenizer, pixel_values, full_question, generation_config)
                        print(f"Assistant: {response}")

                        answers[f"{level}_Q{q_idx + 1}"] = response

                results.append({
                    'Category': category,
                    'Image': image_name,
                    'Prompt': prompt,
                    **answers
                })
                del pixel_values
                torch.cuda.empty_cache()

        all_results.extend(results)
        df = pd.DataFrame(results)
        print(df)
        df.to_csv(output_csv, index=False)

print("All tasks completed.")