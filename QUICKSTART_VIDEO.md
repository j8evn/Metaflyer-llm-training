# 동영상 분석 빠른 시작

동영상을 AI로 분석하여 편집에 활용하는 5단계 가이드입니다.

## 🎬 무엇을 할 수 있나요?

✅ 동영상 → 장면별 이미지 추출  
✅ 음성 → 텍스트 변환 (자막)  
✅ 이미지 내용 분석 (멀티모달 LLM)  
✅ 시각 + 청각 통합 분석  
✅ 편집 가이드 자동 생성  

## 🚀 5단계로 시작하기

### 1단계: 패키지 설치 (5분)

```bash
# ffmpeg 설치 (필수)
brew install ffmpeg  # Mac
# sudo apt install ffmpeg  # Ubuntu

# Python 패키지
pip install -r requirements_video.txt

# 멀티모달 분석용 (선택)
pip install -r requirements_multimodal.txt
```

### 2단계: 동영상 준비 (1분)

```bash
# 동영상 파일 복사
cp your_video.mp4 data/videos/
```

### 3단계: 기본 분석 실행 (5-10분)

```bash
# 자동 분석 스크립트
./scripts/analyze_video.sh data/videos/your_video.mp4
```

또는:

```bash
# 직접 실행
python src/video_analyzer.py data/videos/your_video.mp4 \
    --interval 2.0
```

**출력:**
- 장면별 이미지
- STT 텍스트 (타임스탬프 포함)
- 편집 가이드

### 4단계: 결과 확인 (1분)

```bash
# 편집 가이드 보기
cat outputs/video_analysis/editing_guide.txt

# 분석 결과 보기
cat outputs/video_analysis/analysis_result.json
```

### 5단계: 고급 분석 (선택, 20-30분)

```bash
# 멀티모달 LLM으로 이미지 분석 포함
python src/video_analyzer.py data/videos/your_video.mp4 \
    --interval 2.0 \
    --use_multimodal \
    --multimodal_model_path "llava-hf/llava-1.5-7b-hf"
```

**추가 출력:**
- 각 장면의 시각적 설명
- 분위기/감정 분석
- 주요 객체 인식

---

## 📊 출력 파일 설명

### 1. analysis_result.json (핵심 데이터)

```json
{
  "video_info": {
    "duration": 125.5,
    "fps": 30.0
  },
  "transcript": {
    "text": "전체 대사...",
    "segments": [
      {"start": 0.0, "end": 3.2, "text": "안녕하세요"}
    ]
  },
  "scenes": [
    {
      "scene_number": 1,
      "timestamp": 0.0,
      "frame_path": "frames/frame_001.jpg",
      "dialogue": "안녕하세요",
      "description": "발표자가 등장...",
      "mood": "친근함"
    }
  ]
}
```

### 2. editing_guide.txt (편집 가이드)

```
동영상 편집 가이드
===============================
동영상: my_video.mp4
길이: 125.50초
총 장면: 63개

[장면 1] - 0.00초
  설명: 오프닝 화면
  대사: 안녕하세요, 여러분
  분위기: 친근하고 전문적

[장면 2] - 2.00초
  ...

편집 제안:
  • BGM 추가 권장 (대사 없는 구간)
  • 장면 전환 효과 추가
```

### 3. editing_data.json (편집 SW용)

```json
{
  "markers": [
    {"time": 0.0, "label": "Scene 1", "description": "오프닝"},
    {"time": 2.0, "label": "Scene 2", "description": "본론"}
  ],
  "scenes": [
    {
      "start": 0.0,
      "description": "...",
      "dialogue": "..."
    }
  ]
}
```

---

## 💡 실전 활용

### 유튜브 챕터 자동 생성

```bash
# 1. 동영상 분석
python src/video_analyzer.py youtube_video.mp4

# 2. 챕터 생성
python examples/video_analysis_example.py 5

# 3. 결과를 YouTube 설명란에 복사
00:00 인트로
00:30 주제 소개
01:15 설명 시작
...
```

### 강의 영상 요약

```bash
# 1분마다 캡처하여 주요 슬라이드 추출
python src/video_analyzer.py lecture.mp4 --interval 60.0
```

### 하이라이트 클립 자동 추출

```python
# 중요 장면 자동 감지
from src.video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer("game.mp4")
results = analyzer.analyze_full_pipeline()

# '골', '득점' 등 키워드가 있는 장면 추출
highlights = [
    s for s in results['scenes']
    if any(word in s.get('dialogue', '') 
           for word in ['골', '득점', '와', '대박'])
]

print(f"하이라이트 {len(highlights)}개 발견")
```

---

## 🎓 분석 모드

### 모드 1: 빠른 분석 (STT만)

```bash
python src/video_analyzer.py video.mp4
```

**속도:** ⭐⭐⭐⭐⭐ (5분 영상 = 2-3분)  
**기능:** 프레임 추출 + STT  
**용도:** 자막 생성, 대사 검색

### 모드 2: 완전 분석 (멀티모달 포함)

```bash
python src/video_analyzer.py video.mp4 \
    --use_multimodal \
    --multimodal_model_path "llava-hf/llava-1.5-7b-hf"
```

**속도:** ⭐⭐ (5분 영상 = 15-20분)  
**기능:** 모든 기능  
**용도:** 상세 분석, 자동 편집

---

## 🛠️ 트러블슈팅

### ffmpeg 오류

```bash
# ffmpeg 설치 확인
ffmpeg -version

# 없으면 설치
brew install ffmpeg  # Mac
```

### GPU 메모리 부족

```bash
# Whisper 모델 크기 줄이기
# src/video_analyzer.py에서:
# STTProcessor(model_size="tiny")  # base 대신 tiny

# 또는 CPU 사용
# 자동으로 CPU로 fallback됨
```

### 긴 동영상 처리

```bash
# 간격 늘리기 (프레임 수 감소)
python src/video_analyzer.py long_video.mp4 --interval 5.0

# 또는 분할하여 처리
ffmpeg -i long_video.mp4 -t 300 -c copy part1.mp4
python src/video_analyzer.py part1.mp4
```

---

## 📚 관련 문서

- **VIDEO_ANALYSIS_GUIDE.md** - 완전한 가이드
- **MULTIMODAL_GUIDE.md** - 멀티모달 학습
- **examples/video_analysis_example.py** - Python 예제

---

## 🎉 요약

```bash
# 완전한 워크플로우 (한 줄)
./scripts/analyze_video.sh your_video.mp4 --multimodal

# 결과:
# ✅ 장면별 이미지
# ✅ 전체 자막 (타임스탬프)
# ✅ 각 장면 설명
# ✅ 편집 가이드
```

**동영상 편집이 10배 빨라집니다!** 🎬✨

