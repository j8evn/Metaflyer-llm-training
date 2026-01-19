# LLM 학습 프레임워크 (프로젝트 통합 가이드)

이 가이드는 환경 설정, 학습, 추론을 포함한 LLM 학습 프레임워크에 대한 종합적인 설명을 제공합니다.

---

## 시작하기

### 1. 사전 요구 사항
- Python 3.8 이상
- CUDA 호환 GPU (권장)

### 2. 설정
```bash
./setup.sh
# 또는 수동 설정
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 학습
텍스트 전용 모델과 멀티모달 모델 모두 통합된 스크립트로 학습이 가능합니다.
```bash
# 텍스트 전용 LLM
python src/train.py --config configs/train_config.yaml

# 멀티모달 (비전-언어)
python src/train.py --config configs/multimodal_config.yaml
```

---

## 주요 기능

### 통합 아키텍처
본 프레임워크는 단일 진입점을 통해 표준 인과적 언어 모델(Causal LLM)과 멀티모달(비전-언어) 모델을 모두 지원합니다.

### 고급 최적화 기능
- LoRA / QLoRA: 메모리 효율적인 파인튜닝 지원
- 양자화: 대형 모델을 위한 4비트 및 8비트 지원
- 그래디언트 체크포인팅: 제한된 VRAM 환경에서의 처리량 향상

---

## 상세 설명

### 데이터 준비
데이터는 JSON 형식이어야 합니다.
- 텍스트: `{"instruction": "...", "input": "...", "output": "..."}` 또는 `{"messages": [{"role": "user", "content": "..."}]}`
- 멀티모달: `{"image": "이미지경로", "text": "...", "answer": "..."}`

### 추론
통합 추론 엔진을 사용하십시오.
```bash
# 대화형 채팅
python src/inference.py --model_path 모델경로

# 비전-언어 추론
python src/inference.py --model_path 비전모델경로 --image 이미지.jpg --prompt "이 이미지를 설명해주세요."
```

### API 서비스
추론 서버 시작 방법:
```bash
./scripts/start_api.sh
```

---

## 프로젝트 구조
- src: 핵심 학습 및 추론 로직
- configs: YAML 설정 파일
- scripts: 평가 및 API 관련 유틸리티 스크립트
- data: 데이터셋 및 전처리 스크립트

---

## 유지 관리 및 확장
모델 추가 또는 커스텀 학습 전략에 대한 상세 내용은 docs/MAINTENANCE_GUIDE.md를 참조하십시오.
