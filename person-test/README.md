# Vision-Language Model Fine-tuning Project (Qwen3-VL)

이 프로젝트는 **Qwen3-VL-30B-MoE** 모델을 Fine-tuning하여 특정 인물을 식별하고 설명하는 개인화된 Vision-Language Model을 구축하는 것을 목표로 합니다.

## 디렉토리 구조
## 디렉토리 구조
```text
person-test/
├── config/             # 설정 파일 모음 (인물 리스트 등)
├── data/               # 데이터셋 모음
│   ├── train/          # 학습용 원본 이미지
│   ├── test/           # 검증용 테스트 이미지
│   └── dataset.json    # 정제된 학습 데이터셋 메타데이터
├── scripts/            # 실행 코드 모음
│   ├── data/           # 데이터 수집 및 전처리
│   │   ├── fetch_images.py
│   │   ├── preprocess_data.py
│   │   └── prepare_incremental_dataset.py
│   ├── train/          # 학습
│   │   └── train.py
│   ├── deploy/         # 배포 및 서버 실행
│   │   ├── merge_lora.py
│   │   ├── patch_config.py
│   │   └── start_vllm.sh
│   └── test/           # 테스트
│       ├── automated_test.py
│       └── test_api.py
├── models/             # 결과물 모음
│   ├── weights/        # 학습된 LoRA 어댑터 가중치
│   └── merged/         # vLLM용 최종 병합 모델
└── logs/               # 실행 로그 및 vLLM 로그
```

## 파이프라인 실행 가이드

### 1. 설정 및 데이터 수집
`config/people.json`에 학습할 인물 리스트를 작성한 뒤 데이터를 수집합니다. (1인당 30장 권장) **이미 수집된 데이터는 건너뛰는 증분 수집을 지원합니다.**
```bash
# 1. 학습용 이미지 수집
python scripts/data/fetch_images.py

# 2. 테스트용 이미지 수집 (검증용)
python scripts/data/fetch_test_images.py

# 3. 데이터 정제 및 얼굴 인식/크롭
python scripts/data/preprocess_data.py
```

### 2. 모델 학습 (Training)
H100 GPU를 사용하여 모델을 학습합니다.
```bash
# 백그라운드 학습 실행
nohup python scripts/train/train.py > logs/train.log 2>&1 &
```
*   학습된 LoRA 어댑터는 `models/weights/`에 날짜별 폴더로 저장됩니다.

### 3. 모델 병합 (Merging)
학습된 어댑터를 베이스 모델과 병합하여 배포용 모델을 생성합니다.
```bash
python scripts/deploy/merge_lora.py
```
*   최종 모델은 `models/merged/`에 저장됩니다.

### 4. 서버 서빙 (vLLM)
백그라운드에서 서버를 띄우기 위해 쉘 스크립트를 사용합니다. (기본 Port: 18001)
```bash
chmod +x scripts/deploy/start_vllm.sh
./scripts/deploy/start_vllm.sh
```
*   서버 로그는 `logs/vllm_server.log`에서 실시간으로 확인할 수 있습니다.

### 5. 테스트 및 검증 (Testing)
모델의 식별 성능을 두 가지 방식으로 검증합니다.

#### A. 단일 이미지 콕 집어서 테스트
```bash
python scripts/test/test_api.py data/test/박나래/test_박나래_000.jpg
```

#### B. 전체 정확도 자동 측정 (성적표 출력)
`data/test` 폴더 안의 모든 이미지를 돌려 전체 정확도(%)를 계산하고 상세 보고서를 생성합니다.
```bash
python scripts/test/automated_test.py
```
*   결과 요약은 터미널에 출력되며, 상세 리포트는 `logs/test_report.json`에 저장됩니다.

## 💡 주요 팁
1. **증분 수집**: `people.json`에 인물을 추가하고 `fetch_images.py`를 다시 돌리면, 새로 추가된 사람만 자동으로 수집하고 `dataset.json`을 동기화합니다.
2. **vLLM 포트**: 기본 포트는 `18001`입니다. 변경이 필요하면 `scripts/start_vllm.sh`와 `scripts/test_api.py` 내부의 포트를 함께 수정하세요.
3. **배포 환경**: vLLM 사용 시 `VLLM_USE_V1=0` 환경 변수가 적용되어 MoE 모델 로딩 이슈를 방지합니다.
