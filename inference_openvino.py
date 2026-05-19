import os
import time
import torch
from PIL import Image
import requests

from transformers import AutoProcessor, GenerationConfig, AutoModelForCausalLM

model_path = "./model"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    _attn_implementation='eager'
)

print("Loading generation config...")
generation_config = GenerationConfig.from_pretrained(model_path)

print("\n--- Running inference (CPU) ---")
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

image_url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

image = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(text=prompt, images=image, return_tensors='pt')

print("Generating response...")
start_time = time.time()
generate_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    generation_config=generation_config,
)
end_time = time.time()

generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

inference_time = end_time - start_time
print(f'>>> Response\n{response}')
print(f'\n--- Inference Time: {inference_time:.2f} seconds ---')
print(f'Tokens generated: {len(generate_ids[0])}')
print(f'Tokens per second: {len(generate_ids[0]) / inference_time:.2f}')