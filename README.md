# Phi-4 Multimodal Inference

Inference scripts for Microsoft's Phi-4-multimodal-instruct model.

## Model

[Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) is a 5.6B parameter multimodal model supporting text, image, and audio inputs.

## Requirements

```bash
uv pip install torch transformers accelerate soundfile pillow scipy torchvision backoff peft
uv pip install bitsandbytes  # for 4-bit quantization
uv pip install optimum[openvino]  # for OpenVINO inference
```

## Inference Scripts

### 1. Standard Inference (`inference.py`)
Load model from local directory with full precision.

```bash
uv run python inference.py
```

### 2. 4-bit Quantized Inference (`inference_4bit.py`)
Load model with 4-bit NF4 quantization to reduce memory usage.

```bash
uv run python inference_4bit.py
```

### 3. CPU Inference (`inference_openvino.py`)
Run inference on CPU. Includes inference time measurement.

```bash
uv run python inference_openvino.py
```

## Directory Structure

```
.
├── model/                  # Original model (full precision)
├── model_openvino/        # OpenVINO quantized model
├── model_4bit/            # 4-bit quantized model
├── inference.py           # Standard inference
├── inference_4bit.py      # 4-bit quantized inference
├── inference_openvino.py  # CPU inference with OpenVINO
└── README.md
```

## Model Download

Models should be placed in the appropriate directories:
- `model/` - Download from [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- `model_openvino/` - Download from [Fede2782/Phi-4-multimodal-instruct-int4-cw-ov](https://huggingface.co/Fede2782/Phi-4-multimodal-instruct-int4-cw-ov)

## Usage Example

```python
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import requests

model_path = "./model"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).cuda()
generation_config = GenerationConfig.from_pretrained(model_path)

prompt = '<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>'
image = Image.open(requests.get("image_url").raw)
inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

generate_ids = model.generate(**inputs, max_new_tokens=100, generation_config=generation_config)
response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)
```

## Results & Findings

### Memory Requirements
- **Full precision (FP16)**: Requires ~12GB GPU memory
- **4-bit quantization**: Requires ~6GB GPU memory (tested but needs more than 3.6GB available)
- **CPU inference**: Possible but very slow (not recommended for production)

### Issues Encountered

1. **4-bit Quantization (bitsandbytes)**
   - Failed on GPUs with < 6GB VRAM
   - Error: `CUDA out of memory` even with NF4 quantization
   - The multimodal model (vision + audio encoders) adds overhead beyond the language model

2. **OpenVINO Model**
   - The quantized OpenVINO model (`Fede2782/Phi-4-multimodal-instruct-int4-cw-ov`) has separate components:
     - `openvino_language_model.xml` - Language model
     - `openvino_vision_embeddings_model.xml` - Vision encoder
     - `openvino_audio_embeddings_model.xml` - Audio encoder
   - Loading via `optimum.intel` OVModelForCausalLM fails due to missing `openvino_model.xml`
   - Requires custom loading approach for each component

3. **Small GPU Limitations**
   - This model (5.6B params) is too large for consumer GPUs with < 8GB VRAM
   - Recommended: NVIDIA A100, H100, or equivalent with 24GB+ VRAM

### Recommendations

- For inference: Use cloud GPUs (A100/H100) or quantized versions via vLLM
- For local: Minimum 16GB VRAM recommended
- For 4-bit: Requires ~6GB but multimodal components may need more

