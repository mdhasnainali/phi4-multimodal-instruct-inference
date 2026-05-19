import requests
import torch
import os
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig

model_path = "./model"
quantized_model_path = "./model_4bit"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
    _attn_implementation='eager'
).cuda()

print("Loading generation config...")
generation_config = GenerationConfig.from_pretrained(model_path)

print(f"Saving quantized model to {quantized_model_path}...")
os.makedirs(quantized_model_path, exist_ok=True)
model.save_pretrained(quantized_model_path)
processor.save_pretrained(quantized_model_path)
generation_config.save_pretrained(quantized_model_path)

print("Done! Quantized model saved.")

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

print("\n--- IMAGE PROCESSING ---")
image_url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

image = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')