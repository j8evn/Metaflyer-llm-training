# 멀티모달 모델 빠른 시작

이미지를 이용한 LLM 파인튜닝을 5단계로 시작하는 가이드입니다.

## 🎨 멀티모달 모델이란?

이미지와 텍스트를 함께 이해하는 AI 모델입니다:
- 📸 이미지를 보고 설명 생성
- ❓ 이미지에 대한 질문에 답변
- 🔍 이미지 내용 분석

## 🚀 5단계로 시작하기

### 1단계: 추가 패키지 설치 (2분)

```bash
# 멀티모달용 추가 패키지
pip install -r requirements_multimodal.txt
```

### 2단계: 이미지 데이터 준비 (5분)

```bash
# 이미지 디렉토리 생성
mkdir -p data/images

# 이미지 파일을 data/images/ 에 복사
# 예: cat.jpg, dog.jpg, food.jpg 등
```

샘플 이미지 다운로드:
- Unsplash: https://unsplash.com/
- Pexels: https://www.pexels.com/

### 3단계: JSON 메타데이터 생성 (3분)

```json
// data/multimodal_train.json
[
    {
        "image": "data/images/cat.jpg",
        "question": "이 이미지에 무엇이 있나요?",
        "answer": "고양이가 소파에 앉아 있습니다."
    },
    {
        "image": "data/images/dog.jpg",
        "text": "골든 리트리버 강아지가 공원에서 놀고 있습니다."
    }
]
```

또는 제공된 샘플 사용:
```bash
# 이미 생성되어 있음
cat data/multimodal_train.json
```

### 4단계: 학습 실행 (1-3시간)

```bash
python src/train_multimodal.py \
    --model_name "llava-hf/llava-1.5-7b-hf" \
    --dataset_path "data/multimodal_train.json" \
    --output_dir "outputs/llava_custom"
```

또는 설정 파일 사용:
```bash
python src/train_multimodal.py --config configs/multimodal_config.yaml
```

### 5단계: 추론 테스트 (1분)

```bash
# 이미지 설명
python src/inference_multimodal.py \
    --model_path "outputs/llava_custom/final_model" \
    --model_type "llava" \
    --image "test_image.jpg"

# 이미지 Q&A
python src/inference_multimodal.py \
    --model_path "outputs/llava_custom/final_model" \
    --model_type "llava" \
    --image "test_image.jpg" \
    --question "이 이미지에서 무엇을 볼 수 있나요?"

# 대화형 모드
python src/inference_multimodal.py \
    --model_path "outputs/llava_custom/final_model" \
    --model_type "llava"
```

---

## 📊 지원 모델

### LLaVA (권장) ⭐⭐⭐⭐⭐

```bash
# LLaVA 1.5 7B
--model_name "llava-hf/llava-1.5-7b-hf"

# LLaVA 1.5 13B (더 강력)
--model_name "llava-hf/llava-1.5-13b-hf"
```

**장점:** GPT-4V와 유사한 성능, 오픈 소스

### BLIP-2

```bash
# BLIP-2 OPT 2.7B (가벼움)
--model_name "Salesforce/blip2-opt-2.7b" --model_type "blip2"
```

**장점:** 적은 메모리, 빠른 학습

---

## 💡 실전 예제

### 의료 영상 분석

```json
// data/medical_images.json
[
    {
        "image": "data/images/xray_chest.jpg",
        "question": "이 X-ray에서 이상 소견이 있나요?",
        "answer": "정상 흉부 X-ray입니다. 폐와 심장이 정상 범위 내에 있으며 특이 소견은 관찰되지 않습니다."
    }
]
```

학습:
```bash
python src/train_multimodal.py \
    --model_name "llava-hf/llava-1.5-7b-hf" \
    --dataset_path "data/medical_images.json" \
    --output_dir "outputs/medical_vision_assistant"
```

### 제품 설명 생성

```json
// data/product_images.json
[
    {
        "image": "data/images/product_001.jpg",
        "instruction": "이 제품의 매력적인 판매 문구를 작성하세요",
        "output": "프리미엄 가죽 지갑 - 장인 정신이 담긴 고급스러운 디자인. 실용성과 스타일을 동시에 충족시키는 완벽한 선택입니다."
    }
]
```

### 차트 분석

```json
// data/chart_images.json
[
    {
        "image": "data/images/sales_chart.jpg",
        "question": "이 판매 그래프의 주요 트렌드를 설명하세요",
        "answer": "2023년 초반부터 판매량이 꾸준히 증가하는 추세입니다. 특히 3월과 6월에 급격한 상승이 있었으며, 전년 대비 평균 25% 증가했습니다."
    }
]
```

---

## ⚡ 메모리 최적화

멀티모달 모델은 메모리를 많이 사용합니다!

### GPU 메모리 부족 시

```yaml
# configs/multimodal_config.yaml

# 1. LoRA 사용 (필수)
lora:
  use_lora: true
  r: 16

# 2. 작은 배치
training:
  batch_size: 1
  gradient_accumulation_steps: 16

# 3. 그래디언트 체크포인팅
advanced:
  gradient_checkpointing: true

# 4. 양자화
quantization:
  use_quantization: true
  bits: 4
```

### 시스템 요구사항

| 모델 | 최소 GPU (LoRA) | 권장 GPU |
|------|-----------------|----------|
| BLIP-2 2.7B | 8GB | 12GB |
| LLaVA 1.5 7B | 16GB | 24GB |
| LLaVA 1.5 13B | 24GB | 40GB |

---

## 🎯 전체 워크플로우

```bash
# 1. 패키지 설치
pip install -r requirements.txt
pip install -r requirements_multimodal.txt

# 2. 데이터 준비
mkdir -p data/images
# 이미지 복사
cp ~/Pictures/*.jpg data/images/

# 3. 메타데이터 생성
# data/multimodal_train.json 편집

# 4. 학습
python src/train_multimodal.py --config configs/multimodal_config.yaml

# 5. 추론
python src/inference_multimodal.py \
    --model_path "outputs/multimodal_checkpoints/final_model" \
    --image "test.jpg"
```

---

## 📚 참고 문서

- **MULTIMODAL_GUIDE.md** - 완전한 가이드
- **examples/multimodal_example.py** - Python 예제
- **configs/multimodal_config.yaml** - 설정 파일

---

## 🎓 다음 단계

1. ✅ 이미지 데이터 수집
2. ✅ JSON 메타데이터 생성
3. ✅ 학습 실행
4. ✅ 성능 평가
5. ✅ API 서버 통합

**이미지와 텍스트를 함께 이해하는 AI를 만들어보세요!** 🎨🤖


