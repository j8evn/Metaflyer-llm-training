# 멀티모달 모델 파인튜닝 가이드

이미지와 텍스트를 함께 사용하는 Vision-Language 모델 파인튜닝 가이드입니다.

## 목차
1. [멀티모달 모델이란?](#멀티모달-모델이란)
2. [지원 모델](#지원-모델)
3. [데이터 준비](#데이터-준비)
4. [학습 실행](#학습-실행)
5. [추론 및 사용](#추론-및-사용)

---

## 멀티모달 모델이란?

**Vision-Language 모델**은 이미지와 텍스트를 함께 이해하는 AI 모델입니다.

### 주요 기능

- 📸 **이미지 설명** (Image Captioning)
- ❓ **이미지 Q&A** (Visual Question Answering)
- 🔍 **이미지 기반 대화**
- 📊 **차트/그래프 분석**
- 📝 **OCR 및 문서 이해**

### 사용 사례

- 의료 영상 분석
- 제품 이미지 설명 생성
- 시각적 콘텐츠 검색
- 자동 이미지 태깅
- 시각 장애인 보조

---

## 지원 모델

### LLaVA (Large Language and Vision Assistant)

**가장 인기 있는 오픈 소스 Vision-Language 모델**

```bash
# LLaVA 1.5 7B
model_name: "llava-hf/llava-1.5-7b-hf"

# LLaVA 1.5 13B
model_name: "llava-hf/llava-1.5-13b-hf"

# LLaVA 1.6 (Mistral 기반)
model_name: "llava-hf/llava-v1.6-mistral-7b-hf"
```

**특징:**
- ✅ GPT-4V와 유사한 성능
- ✅ 오픈 소스
- ✅ 파인튜닝 용이

### BLIP-2

```bash
# BLIP-2 OPT 2.7B
model_name: "Salesforce/blip2-opt-2.7b"

# BLIP-2 Flan-T5 XL
model_name: "Salesforce/blip2-flan-t5-xl"
```

**특징:**
- ✅ 효율적인 학습
- ✅ 다양한 백본 지원
- ✅ 빠른 추론

### InstructBLIP

```bash
model_name: "Salesforce/instructblip-vicuna-7b"
```

**특징:**
- ✅ Instruction following
- ✅ 복잡한 질문 처리

---

## 데이터 준비

### 데이터 형식

멀티모달 학습 데이터는 **이미지 파일 + JSON 메타데이터** 형식입니다.

#### 형식 1: 이미지 설명 (Captioning)

```json
[
    {
        "image": "data/images/cat.jpg",
        "text": "고양이가 소파에 편안하게 앉아 있습니다."
    },
    {
        "image": "data/images/dog.jpg",
        "text": "골든 리트리버 강아지가 공원에서 뛰어놀고 있습니다."
    }
]
```

#### 형식 2: Visual Q&A

```json
[
    {
        "image": "data/images/food.jpg",
        "question": "이 음식은 무엇인가요?",
        "answer": "맛있어 보이는 피자입니다. 토핑으로 치즈, 토마토, 바질이 올라가 있습니다."
    },
    {
        "image": "data/images/chart.jpg",
        "question": "이 그래프에서 가장 높은 값은?",
        "answer": "2023년 3월이 가장 높은 값으로 약 150을 나타내고 있습니다."
    }
]
```

#### 형식 3: Instruction 형식

```json
[
    {
        "image": "data/images/product.jpg",
        "instruction": "이 제품의 특징을 설명하세요",
        "input": "",
        "output": "이 제품은 고급 가죽으로 만들어진 검은색 지갑입니다. 세련된 디자인과 실용적인 카드 슬롯이 특징입니다."
    }
]
```

### 디렉토리 구조

```
data/
├── images/              # 이미지 파일들
│   ├── cat.jpg
│   ├── dog.jpg
│   ├── food.jpg
│   └── ...
├── multimodal_train.json  # 학습 메타데이터
└── multimodal_eval.json   # 평가 메타데이터
```

### 샘플 데이터 생성

```python
# 샘플 JSON 생성
python -c "from src.multimodal_utils import create_sample_multimodal_dataset; \
create_sample_multimodal_dataset('data/multimodal_train.json', 20)"
```

**주의:** 실제 이미지 파일을 `data/images/` 디렉토리에 준비해야 합니다!

---

## 학습 실행

### 기본 학습

```bash
python src/train_multimodal.py \
    --model_name "llava-hf/llava-1.5-7b-hf" \
    --model_type "llava" \
    --dataset_path "data/multimodal_train.json" \
    --output_dir "outputs/llava_finetuned"
```

### 설정 파일 사용

```bash
python src/train_multimodal.py --config configs/multimodal_config.yaml
```

### LoRA를 사용한 효율적 학습

```bash
# configs/multimodal_config.yaml
lora:
  use_lora: true
  r: 16
  lora_alpha: 32

training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

실행:
```bash
python src/train_multimodal.py --config configs/multimodal_config.yaml
```

---

## 추론 및 사용

### 1. 이미지 설명 생성

```bash
python src/inference_multimodal.py \
    --model_path "outputs/llava_finetuned/final_model" \
    --model_type "llava" \
    --image "test_image.jpg"
```

### 2. 이미지에 대한 질문

```bash
python src/inference_multimodal.py \
    --model_path "outputs/llava_finetuned/final_model" \
    --model_type "llava" \
    --image "test_image.jpg" \
    --question "이 이미지에서 무엇을 볼 수 있나요?"
```

### 3. 커스텀 프롬프트

```bash
python src/inference_multimodal.py \
    --model_path "outputs/llava_finetuned/final_model" \
    --model_type "llava" \
    --image "test_image.jpg" \
    --prompt "이 이미지의 색상 구성을 분석하세요"
```

### 4. 대화형 모드

```bash
python src/inference_multimodal.py \
    --model_path "outputs/llava_finetuned/final_model" \
    --model_type "llava"
```

대화형 인터페이스:
```
> describe data/images/cat.jpg
설명:
귀여운 고양이가 소파에서 낮잠을 자고 있습니다...

> ask data/images/food.jpg 이 음식은 무엇인가요?
답변:
맛있어 보이는 피자입니다...
```

---

## Python 코드로 사용

```python
from src.multimodal_utils import MultiModalModel

# 모델 초기화
model = MultiModalModel(
    model_name="outputs/llava_finetuned/final_model",
    model_type="llava"
)

# 이미지 설명
description = model.generate_from_image(
    image_path="test_image.jpg",
    prompt="이 이미지를 자세히 설명해주세요.",
    max_new_tokens=256
)

print(description)
```

---

## 실전 예제

### 예제 1: 의료 영상 분석

```json
// data/medical_train.json
[
    {
        "image": "data/images/xray_001.jpg",
        "question": "이 X-ray 이미지에서 이상 소견이 있나요?",
        "answer": "좌측 폐 하부에 약간의 음영이 관찰됩니다. 추가 검사가 필요할 수 있습니다."
    }
]
```

학습:
```bash
python src/train_multimodal.py \
    --model_name "llava-hf/llava-1.5-7b-hf" \
    --dataset_path "data/medical_train.json" \
    --output_dir "outputs/medical_assistant"
```

### 예제 2: 제품 설명 생성

```json
// data/product_train.json
[
    {
        "image": "data/images/product_001.jpg",
        "instruction": "이 제품의 판매 문구를 작성하세요",
        "output": "프리미엄 가죽 지갑 - 세련된 디자인과 뛰어난 내구성을 자랑하는 최고급 제품입니다. 12개의 카드 슬롯과 넉넉한 지폐 공간을 제공합니다."
    }
]
```

### 예제 3: 차트 분석

```json
// data/chart_train.json
[
    {
        "image": "data/images/sales_chart.jpg",
        "question": "이 판매 그래프의 주요 트렌드는?",
        "answer": "2023년 1월부터 6월까지 매출이 꾸준히 상승하는 추세를 보이고 있으며, 특히 3월에 급격한 증가가 있었습니다."
    }
]
```

---

## 메모리 최적화

멀티모달 모델은 메모리를 많이 사용합니다. 최적화 팁:

### 1. LoRA 사용 (필수)

```yaml
lora:
  use_lora: true
  r: 16
```

### 2. 작은 배치 크기

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

### 3. 그래디언트 체크포인팅

```yaml
advanced:
  gradient_checkpointing: true
```

### 4. 이미지 크기 조정

```yaml
image_processing:
  resize: [224, 224]  # 작게 설정
```

---

## 지원 모델 상세

### LLaVA 계열

| 모델 | 크기 | GPU 메모리 (LoRA) | 특징 |
|------|------|-------------------|------|
| llava-1.5-7b-hf | 7B | 16GB | 균형잡힌 |
| llava-1.5-13b-hf | 13B | 24GB | 고성능 |
| llava-v1.6-mistral-7b | 7B | 16GB | Mistral 기반 |

### BLIP 계열

| 모델 | 크기 | GPU 메모리 | 특징 |
|------|------|------------|------|
| blip2-opt-2.7b | 2.7B | 8GB | 가벼움 |
| blip2-flan-t5-xl | 3B | 12GB | T5 기반 |

---

## 전체 워크플로우

```bash
# 1. 추가 의존성 설치
pip install -r requirements_multimodal.txt

# 2. 이미지 데이터 준비
mkdir -p data/images
# 이미지 파일들을 data/images/에 복사

# 3. JSON 메타데이터 생성
cat > data/multimodal_train.json << 'JSON'
[
    {
        "image": "data/images/image1.jpg",
        "question": "이 이미지를 설명하세요",
        "answer": "상세한 설명..."
    }
]
JSON

# 4. 학습
python src/train_multimodal.py --config configs/multimodal_config.yaml

# 5. 추론
python src/inference_multimodal.py \
    --model_path "outputs/multimodal_checkpoints/final_model" \
    --model_type "llava" \
    --image "test.jpg"
```

---

## 데이터셋 예제

### 공개 데이터셋 사용

#### COCO Captions

```python
from datasets import load_dataset

# COCO 데이터셋 로딩
dataset = load_dataset("HuggingFaceM4/COCO")

# 변환
multimodal_data = []
for item in dataset['train']:
    multimodal_data.append({
        "image": item['image'],  # PIL Image
        "text": item['sentences'][0]['raw']
    })
```

#### VQA (Visual Question Answering)

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceM4/VQAv2")

for item in dataset['train']:
    multimodal_data.append({
        "image": item['image'],
        "question": item['question'],
        "answer": item['multiple_choice_answer']
    })
```

---

## Python 사용 예제

### 학습된 모델로 추론

```python
from src.inference_multimodal import MultiModalInference

# 엔진 초기화
engine = MultiModalInference(
    model_path="outputs/llava_finetuned/final_model",
    model_type="llava"
)

# 이미지 설명
description = engine.describe_image("cat.jpg")
print(f"설명: {description}")

# 이미지 Q&A
answer = engine.answer_question(
    image_path="chart.jpg",
    question="이 차트의 트렌드는?"
)
print(f"답변: {answer}")

# 커스텀 프롬프트
result = engine.generate(
    image_path="product.jpg",
    prompt="이 제품의 장단점을 분석하세요",
    max_new_tokens=300
)
print(f"분석: {result}")
```

---

## API 서버에 통합

### 멀티모달 API 엔드포인트 추가

```python
# src/api_server.py에 추가

from fastapi import File, UploadFile
from src.multimodal_utils import MultiModalModel
import shutil

# 멀티모달 모델 초기화
multimodal_model = None

@app.post("/multimodal/describe")
async def describe_image(file: UploadFile = File(...)):
    """이미지 설명 생성"""
    
    # 임시 파일 저장
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 설명 생성
        description = multimodal_model.generate_from_image(
            temp_path,
            "이 이미지를 설명해주세요."
        )
        
        return {"description": description}
    
    finally:
        # 임시 파일 삭제
        os.remove(temp_path)


@app.post("/multimodal/vqa")
async def visual_qa(
    file: UploadFile = File(...),
    question: str = ""
):
    """Visual Question Answering"""
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        answer = multimodal_model.generate_from_image(
            temp_path,
            f"질문: {question}\n답변:"
        )
        
        return {
            "question": question,
            "answer": answer
        }
    
    finally:
        os.remove(temp_path)
```

사용:
```bash
# 이미지 설명
curl -X POST http://localhost:8000/multimodal/describe \
  -F "file=@cat.jpg"

# Visual Q&A
curl -X POST "http://localhost:8000/multimodal/vqa?question=무엇이보이나요" \
  -F "file=@image.jpg"
```

---

## 베스트 프랙티스

### 1. 데이터 품질

✅ **좋은 데이터:**
- 고해상도 이미지 (최소 224x224)
- 상세하고 정확한 설명
- 다양한 각도와 조명

❌ **피해야 할 것:**
- 흐릿하거나 저화질 이미지
- 부정확한 설명
- 편향된 데이터

### 2. 하이퍼파라미터

```yaml
# 추천 설정
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5

lora:
  r: 16  # Vision-Language는 16 추천
  lora_alpha: 32
```

### 3. 평가

```python
# 정성 평가
from src.inference_multimodal import MultiModalInference

engine = MultiModalInference("outputs/model", "llava")

test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]

for img in test_images:
    desc = engine.describe_image(img)
    print(f"이미지: {img}")
    print(f"설명: {desc}\n")
```

---

## 트러블슈팅

### 문제 1: GPU 메모리 부족

```yaml
# 해결책
training:
  batch_size: 1
  gradient_accumulation_steps: 16

lora:
  use_lora: true

quantization:
  use_quantization: true
  bits: 4
```

### 문제 2: 이미지 로딩 실패

```python
# 이미지 경로 확인
import os
print(os.path.exists("data/images/cat.jpg"))

# PIL 이미지 테스트
from PIL import Image
img = Image.open("data/images/cat.jpg")
img.show()
```

### 문제 3: 학습 속도 느림

- 이미지 크기 줄이기 (224x224)
- 배치 크기 조정
- 데이터 로더 워커 수 증가

---

## 고급: 커스텀 Vision Encoder

자신만의 vision encoder 사용:

```python
from transformers import CLIPVisionModel

# CLIP vision encoder
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

# LLM과 결합
# (고급 사용자용 - 아키텍처 수정 필요)
```

---

## 요약

### 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements_multimodal.txt

# 2. 데이터 준비
# data/images/ 에 이미지 파일
# data/multimodal_train.json 에 메타데이터

# 3. 학습
python src/train_multimodal.py --config configs/multimodal_config.yaml

# 4. 추론
python src/inference_multimodal.py \
    --model_path "outputs/multimodal_checkpoints/final_model" \
    --image "test.jpg"
```

### 지원 모델

- ✅ LLaVA (권장)
- ✅ BLIP-2
- ✅ InstructBLIP

### 데이터 형식

- Image Captioning: image + text
- Visual Q&A: image + question + answer
- Instruction: image + instruction + output

**멀티모달 AI의 세계에 오신 것을 환영합니다!** 🎨🤖


