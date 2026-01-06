import string
import torch
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig
from PIL import Image
import requests
import json
import random
import base64
from io import BytesIO
from requests.exceptions import Timeout, RequestException
import time


def parse_sectioned_prompt(s):
    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result

def ds2(args, device, model, tokenizer, data, temperature=0.8, n=1, top_p=0.7, max_tokens=256, batch_size=16):
    outputs = []
    if 'internvl' in args['LLM'].lower() or 'llama' in args['LLM'].lower():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            images, texts = zip(*batch)
            messages_ = []
            for image, text in zip(images, texts):
                if isinstance(image, Image.Image):
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ]}
                    ]
                elif isinstance(image, (list, tuple)) and all(isinstance(img, Image.Image) for img in image):
                    messages_content = [{"type": "image", "image": img} for img in image]
                    messages_content.append({"type": "text", "text": text})
                    messages = [{"role": "user", "content": messages_content}]
                else:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": text},
                        ]}
                    ]
                messages_.append(messages)
            inputs = tokenizer.apply_chat_template(
                messages_, tokenize=True, add_generation_prompt=True, padding=True, return_dict=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                if 'internvl' in args['LLM'].lower():
                    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, pad_token_id=tokenizer.tokenizer.eos_token_id)
                else:
                    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                                         zip(inputs.input_ids, generated_ids)]
            output = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            print("output:", output)
            outputs.extend(output)

            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()
    elif 'gpt' in args['LLM'].lower() or 'gemini' in args['LLM'].lower():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            images, texts = zip(*batch)
            output = chat(args['provider'], texts, images, max_tokens)
            print("output:", output)
            outputs.append(output)

    else:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            messages_batch = []
            for image, text in batch:
                if isinstance(image, Image.Image):
                    content = [{"type": "image", "image": image},
                               {"type": "text", "text": text}]
                elif isinstance(image, (list, tuple)):
                    content = [{"type": "image", "image": img} for img in image]
                    content.append({"type": "text", "text": text})
                else:
                    content = [{"type": "text", "text": text}]

                messages_batch.append([{"role": "user", "content": content}])

            texts_ = tokenizer.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
            images_, _ = process_vision_info(messages_batch)

            if images_ and any(img is not None for img in images_):
                inputs = tokenizer(
                    text=texts_,
                    images=images_,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device, non_blocking=True)
            else:
                inputs = tokenizer(
                    text=texts_,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device, non_blocking=True)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True,clean_up_tokenization_spaces=False)
            print("output:", output)
            outputs.extend(output)

            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()
    print(outputs)
    return outputs

def chat(provider, texts, images=None, max_tokens=128, timeout=12, max_retries=50):
    if not isinstance(texts, str):
        texts = texts[0] if isinstance(texts, list) else str(texts)
    clean_images = []
    if images:
        for img in images:
            if isinstance(img, list):
                for sub_img in img:
                    if isinstance(sub_img, Image.Image):
                        clean_images.append(sub_img)
            elif isinstance(img, Image.Image):
                clean_images.append(img)
    images = clean_images
    api_key = ['']
    url = ""
    model = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash-lite",
    }[provider]
    messages = build_messages(texts, images)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False
    }

    for attempt in range(max_retries):
        key = random.choice(api_key)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )

            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429 or resp.status_code == 500:
                print(f"[Attempt {attempt + 1}] HTTP {resp.status_code}: {resp.text}")
                break
            else:
                print(f"[Attempt {attempt + 1}] HTTP {resp.status_code}: {resp.text}")
                time.sleep(2)

        except Timeout:
            print(f"[Attempt {attempt + 1}] Request timeout (> {timeout}s), retrying...")
            time.sleep(2 ** attempt)
        except RequestException as e:
            print(f"[Attempt {attempt + 1}] Request error: {e}")
            time.sleep(2 ** attempt)
        time.sleep(1.0) 
    return ""

def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def build_messages(texts, images):
    content = [{"type": "text", "text": texts}]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + pil_to_base64(img)
            }
        })
    return [{
        "role": "user",
        "content": content
    }]



