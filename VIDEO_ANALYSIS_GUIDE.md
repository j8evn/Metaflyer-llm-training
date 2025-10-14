# 동영상 분석 가이드

동영상을 장면별로 분석하여 편집에 활용하는 완전한 가이드입니다.

## 📹 시스템 개요

```
동영상 파일
    ↓
┌─────────────┐     ┌──────────────┐
│ 시각 (영상) │     │ 청각 (음성)  │
│  ffmpeg     │     │   Whisper    │
│  ↓ 이미지   │     │   ↓ 텍스트   │
└─────────────┘     └──────────────┘
    ↓                      ↓
    └──────────┬───────────┘
               ↓
       멀티모달 LLM 분석
               ↓
       종합 분석 결과
               ↓
    동영상 편집 메타데이터
```

## 🎯 주요 기능

1. **장면 추출**: ffmpeg로 동영상 → 이미지
2. **음성 인식**: Whisper로 음성 → 텍스트
3. **시각 분석**: 멀티모달 LLM으로 이미지 분석
4. **종합 분석**: 이미지 + 대사 통합 분석
5. **편집 가이드**: 자동 편집 제안 생성

---

## 🚀 빠른 시작

### 1단계: 의존성 설치

```bash
# 비디오 처리 패키지
pip install -r requirements_video.txt

# 멀티모달 모델 (선택, 고급 분석용)
pip install -r requirements_multimodal.txt

# ffmpeg 설치 (필수)
# Mac
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# 확인
ffmpeg -version
```

### 2단계: 기본 분석 (STT만)

```bash
python src/video_analyzer.py your_video.mp4
```

출력:
- `outputs/video_analysis/frames/` - 추출된 이미지들
- `outputs/video_analysis/audio.wav` - 추출된 오디오
- `outputs/video_analysis/analysis_result.json` - 분석 결과
- `outputs/video_analysis/editing_guide.txt` - 편집 가이드

### 3단계: 고급 분석 (멀티모달 포함)

```bash
python src/video_analyzer.py your_video.mp4 \
    --use_multimodal \
    --multimodal_model_path "llava-hf/llava-1.5-7b-hf"
```

---

## 📝 상세 사용법

### 프레임 추출 방식

#### 방법 1: 일정 간격으로 추출 (권장)

```bash
# 2초마다 프레임 추출
python src/video_analyzer.py video.mp4 --interval 2.0

# 1초마다 (더 자세한 분석)
python src/video_analyzer.py video.mp4 --interval 1.0

# 5초마다 (빠른 분석)
python src/video_analyzer.py video.mp4 --interval 5.0
```

#### 방법 2: 장면 변화 감지

```python
from src.video_processing import VideoProcessor

processor = VideoProcessor("video.mp4")

# 장면 변화 감지로 추출
frames = processor.extract_frames_by_scene(
    threshold=30.0,  # 높을수록 큰 변화만 감지
    min_scene_duration=1.0  # 최소 장면 길이
)
```

### STT (Speech-to-Text)

```python
from src.stt_utils import STTProcessor

# STT 프로세서 초기화
stt = STTProcessor(
    model_size="base",  # tiny, base, small, medium, large
    language="ko"  # 한국어
)

# 음성 인식
result = stt.transcribe("audio.wav")

print(f"전체 텍스트: {result['text']}")

# 타임스탬프별 대사
for seg in result['segments']:
    print(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")
```

### 멀티모달 분석

```python
from src.multimodal_utils import MultiModalModel

# 모델 로딩
model = MultiModalModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="llava"
)

# 이미지 분석
description = model.generate_from_image(
    "frame_001.jpg",
    "이 장면을 설명하세요"
)

# 감정 분석
mood = model.generate_from_image(
    "frame_001.jpg",
    "이 장면의 분위기는?"
)
```

---

## 🎬 완전한 파이프라인

### Python 스크립트

```python
# analyze_video.py
from src.video_analyzer import VideoAnalyzer
from src.multimodal_utils import MultiModalModel

# 1. 멀티모달 모델 로딩 (선택)
multimodal_model = MultiModalModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="llava"
)

# 2. 분석기 초기화
analyzer = VideoAnalyzer(
    video_path="my_video.mp4",
    multimodal_model=multimodal_model,
    output_dir="outputs/my_analysis"
)

# 3. 전체 분석 실행
results = analyzer.analyze_full_pipeline(
    extract_method="interval",
    interval_seconds=2.0
)

# 4. 편집 가이드 생성
guide = analyzer.generate_editing_guide(results)
print(guide)

# 5. 편집 소프트웨어용 데이터 export
analyzer.export_for_editing(results, format="json")
```

### 커맨드 라인

```bash
# 기본 분석 (STT + 프레임 추출)
python src/video_analyzer.py video.mp4

# 멀티모달 분석 포함
python src/video_analyzer.py video.mp4 \
    --use_multimodal \
    --multimodal_model_path "llava-hf/llava-1.5-7b-hf"

# 간격 조정
python src/video_analyzer.py video.mp4 \
    --interval 1.0 \
    --use_multimodal \
    --multimodal_model_path "outputs/my_model"
```

---

## 📊 출력 데이터 형식

### analysis_result.json

```json
{
  "video_info": {
    "fps": 30.0,
    "duration": 120.5,
    "width": 1920,
    "height": 1080
  },
  "frames": [
    "outputs/video_analysis/frames/frame_000001_t0.00s.jpg",
    "outputs/video_analysis/frames/frame_000002_t2.00s.jpg"
  ],
  "transcript": {
    "text": "전체 대사...",
    "segments": [
      {
        "start": 0.0,
        "end": 3.5,
        "text": "안녕하세요, 여러분"
      }
    ]
  },
  "scenes": [
    {
      "scene_number": 1,
      "timestamp": 0.0,
      "frame_path": "outputs/video_analysis/frames/frame_000001_t0.00s.jpg",
      "dialogue": "안녕하세요, 여러분",
      "description": "발표자가 화면 중앙에 서 있습니다...",
      "mood": "전문적이고 친근한 분위기",
      "objects": "사람, 마이크, 프레젠테이션 화면"
    }
  ]
}
```

### editing_data.json (편집 소프트웨어용)

```json
{
  "video": "my_video.mp4",
  "markers": [
    {
      "time": 0.0,
      "label": "Scene 1",
      "description": "오프닝 장면"
    }
  ],
  "scenes": [
    {
      "start": 0.0,
      "description": "발표자 등장...",
      "dialogue": "안녕하세요...",
      "mood": "친근함"
    }
  ]
}
```

---

## 🎯 실전 활용 사례

### 사례 1: YouTube 동영상 자동 챕터 생성

```python
from src.video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer("youtube_video.mp4")
results = analyzer.analyze_full_pipeline(interval_seconds=30.0)

# 챕터 생성
chapters = []
for scene in results['scenes']:
    timestamp = scene['timestamp']
    description = scene.get('description', '')[:50]
    
    # YouTube 타임스탬프 형식
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    chapters.append(f"{minutes:02d}:{seconds:02d} - {description}")

# YouTube 설명란에 붙여넣기
print("YouTube 챕터:")
for chapter in chapters:
    print(chapter)
```

출력:
```
00:00 - 인트로: 발표자 소개
00:30 - 주제 설명: AI 기술 개요
01:15 - 데모 시연: 실제 사용 예제
02:30 - 질의응답 시작
```

### 사례 2: 강의 동영상 요약

```python
from src.video_analyzer import VideoAnalyzer
from src.multimodal_utils import MultiModalModel

# 멀티모달 모델
mm_model = MultiModalModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="llava"
)

# 분석
analyzer = VideoAnalyzer("lecture.mp4", multimodal_model=mm_model)
results = analyzer.analyze_full_pipeline(interval_seconds=60.0)

# 강의 요약 생성
summary = {
    '제목': '강의 제목',
    '길이': f"{results['video_info']['duration'] / 60:.1f}분",
    '주요 내용': []
}

for scene in results['scenes']:
    if scene.get('description'):
        summary['주요 내용'].append({
            '시간': f"{int(scene['timestamp'] // 60)}:{int(scene['timestamp'] % 60):02d}",
            '내용': scene['description'][:100],
            '대사': scene.get('dialogue', '')[:100]
        })

print(json.dumps(summary, ensure_ascii=False, indent=2))
```

### 사례 3: 동영상 하이라이트 추출

```python
from src.video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer("game_replay.mp4")
results = analyzer.analyze_full_pipeline()

# 중요 장면 찾기 (큰 소리, 급격한 변화 등)
highlights = []

for scene in results['scenes']:
    dialogue = scene.get('dialogue', '').lower()
    
    # 환호 또는 중요 키워드 감지
    if any(word in dialogue for word in ['와', '오', '대박', '골', '득점']):
        highlights.append({
            'timestamp': scene['timestamp'],
            'reason': '중요 이벤트 감지',
            'dialogue': scene['dialogue']
        })

print(f"하이라이트 {len(highlights)}개 발견:")
for h in highlights:
    print(f"  {h['timestamp']:.2f}s - {h['reason']}: {h['dialogue'][:50]}")
```

### 사례 4: 자막 파일 생성

```python
from src.stt_utils import STTProcessor

# STT
stt = STTProcessor(model_size="medium", language="ko")
transcript = stt.transcribe("video_audio.wav")

# SRT 자막 파일 생성
def generate_srt(segments, output_file):
    """SRT 형식 자막 생성"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            # 타임코드 변환
            start = format_srt_time(seg['start'])
            end = format_srt_time(seg['end'])
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text']}\n\n")

def format_srt_time(seconds):
    """초를 SRT 타임코드로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# 사용
generate_srt(transcript['segments'], "subtitles.srt")
```

---

## 🛠️ 단계별 워크플로우

### 단계 1: 동영상 준비

```bash
# 동영상 파일 복사
cp ~/Videos/my_video.mp4 data/videos/
```

### 단계 2: 기본 분석 (STT만)

```bash
python src/video_analyzer.py data/videos/my_video.mp4 \
    --interval 2.0 \
    --output_dir outputs/my_video_analysis
```

**출력:**
- 프레임 이미지들
- 오디오 파일
- STT 텍스트
- 기본 메타데이터

**소요 시간:** 5-10분 (5분 동영상 기준)

### 단계 3: 멀티모달 분석 (고급)

```bash
python src/video_analyzer.py data/videos/my_video.mp4 \
    --interval 2.0 \
    --use_multimodal \
    --multimodal_model_path "llava-hf/llava-1.5-7b-hf"
```

**출력:**
- 모든 기본 분석 +
- 각 장면의 시각적 설명
- 감정/분위기 분석
- 객체 인식 결과

**소요 시간:** 20-30분 (5분 동영상, GPU 사용)

---

## 📊 활용 예제

### 예제 1: 편집 포인트 찾기

```python
import json

# 분석 결과 로딩
with open('outputs/video_analysis/analysis_result.json', 'r') as f:
    results = json.load(f)

# 편집 포인트 추출
edit_points = []

for scene in results['scenes']:
    timestamp = scene['timestamp']
    
    # 장면 전환 포인트
    edit_points.append({
        'time': timestamp,
        'type': 'scene_change',
        'frame': scene['frame_path']
    })
    
    # 대사 시작 포인트
    if scene.get('dialogue') and scene['dialogue'].strip():
        edit_points.append({
            'time': timestamp,
            'type': 'dialogue_start',
            'text': scene['dialogue'][:50]
        })

# Premiere Pro XML 생성
print("편집 포인트:")
for point in edit_points:
    print(f"{point['time']:.2f}s - {point['type']}")
```

### 예제 2: 장면별 태그 생성

```python
# 분석 결과에서 태그 생성
tags_by_scene = []

for scene in results['scenes']:
    tags = []
    
    # 객체 기반 태그
    if scene.get('objects'):
        objects = scene['objects'].lower()
        if '사람' in objects:
            tags.append('people')
        if '자연' in objects or '풍경' in objects:
            tags.append('nature')
        if '실내' in objects:
            tags.append('indoor')
    
    # 분위기 기반 태그
    if scene.get('mood'):
        mood = scene['mood'].lower()
        if '밝' in mood or '즐거' in mood:
            tags.append('positive')
        if '어두' in mood or '슬' in mood:
            tags.append('serious')
    
    tags_by_scene.append({
        'timestamp': scene['timestamp'],
        'tags': tags
    })

# 활용: 특정 태그가 있는 장면만 편집
nature_scenes = [
    s for s in tags_by_scene
    if 'nature' in s['tags']
]
```

### 예제 3: 자동 하이라이트 영상 생성

```python
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 분석 결과에서 하이라이트 찾기
def find_highlights(results):
    """중요 장면 찾기"""
    highlights = []
    
    for scene in results['scenes']:
        score = 0
        
        # 대사가 있으면 +1
        if scene.get('dialogue'):
            score += 1
        
        # 감정 키워드 있으면 +2
        dialogue = scene.get('dialogue', '').lower()
        if any(word in dialogue for word in ['중요', '핵심', '포인트', '와', '대박']):
            score += 2
        
        # 분위기가 역동적이면 +1
        mood = scene.get('mood', '').lower()
        if any(word in mood for word in ['역동', '흥미', '극적']):
            score += 1
        
        if score >= 2:
            highlights.append({
                'timestamp': scene['timestamp'],
                'score': score,
                'duration': 3.0  # 3초 클립
            })
    
    return highlights

# 하이라이트 클립 생성
video = VideoFileClip("my_video.mp4")
highlights = find_highlights(results)

clips = []
for h in highlights:
    start = h['timestamp']
    end = start + h['duration']
    clip = video.subclip(start, end)
    clips.append(clip)

# 하이라이트 영상 합치기
if clips:
    final = concatenate_videoclips(clips)
    final.write_videofile("highlights.mp4")
    print(f"하이라이트 영상 생성 완료: highlights.mp4")
```

---

## 🎨 고급 기능

### 커스텀 분석 파이프라인

```python
from src.video_processing import VideoProcessor
from src.stt_utils import STTProcessor
from src.multimodal_utils import MultiModalModel

class CustomVideoAnalyzer:
    """커스텀 동영상 분석기"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        
        # 컴포넌트 초기화
        self.video_proc = VideoProcessor(video_path)
        self.stt = STTProcessor(model_size="base", language="ko")
        self.vision_model = MultiModalModel(
            model_name="llava-hf/llava-1.5-7b-hf",
            model_type="llava"
        )
    
    def analyze_for_editing(self):
        """편집용 종합 분석"""
        
        # 1. 장면 감지로 프레임 추출
        frames = self.video_proc.extract_frames_by_scene()
        
        # 2. 오디오 추출 및 STT
        audio = self.video_proc.extract_audio()
        transcript = self.stt.transcribe(audio)
        
        # 3. 각 장면 분석
        analysis = []
        
        for frame_path in frames:
            timestamp = self.video_proc.get_frame_timestamp(
                os.path.basename(frame_path)
            )
            
            # 이미지 분석
            scene_desc = self.vision_model.generate_from_image(
                frame_path,
                "이 장면에서 일어나는 일을 설명하세요"
            )
            
            # 해당 시간 대사
            dialogue = self.stt.get_transcript_at_time(
                transcript['segments'],
                timestamp,
                context_window=5.0
            )
            
            # 편집 제안
            edit_suggestion = self._suggest_edit(scene_desc, dialogue)
            
            analysis.append({
                'timestamp': timestamp,
                'visual': scene_desc,
                'audio': dialogue,
                'edit_suggestion': edit_suggestion
            })
        
        return analysis
    
    def _suggest_edit(self, visual: str, audio: str) -> str:
        """장면 기반 편집 제안"""
        suggestions = []
        
        if not audio.strip():
            suggestions.append("BGM 추가 권장")
        
        if '사람' in visual and len(audio.split()) < 3:
            suggestions.append("클로즈업 샷 고려")
        
        if '풍경' in visual or '자연' in visual:
            suggestions.append("와이드 샷 유지")
        
        return ' | '.join(suggestions) if suggestions else "기본 편집"
```

---

## ⚙️ 설정 옵션

### Whisper 모델 크기 선택

| 모델 | 크기 | 속도 | 정확도 | 용도 |
|------|------|------|--------|------|
| tiny | 39M | ⭐⭐⭐⭐⭐ | ⭐⭐ | 빠른 테스트 |
| base | 74M | ⭐⭐⭐⭐ | ⭐⭐⭐ | **권장** |
| small | 244M | ⭐⭐⭐ | ⭐⭐⭐⭐ | 균형 |
| medium | 769M | ⭐⭐ | ⭐⭐⭐⭐⭐ | 고품질 |
| large | 1550M | ⭐ | ⭐⭐⭐⭐⭐ | 최고 품질 |

### 프레임 추출 간격

| 간격 | 용도 | 프레임 수 (5분 영상) |
|------|------|---------------------|
| 0.5초 | 상세 분석 | 600개 |
| 1.0초 | 일반 분석 | 300개 |
| 2.0초 | **권장** | 150개 |
| 5.0초 | 빠른 분석 | 60개 |
| 10초 | 개요만 | 30개 |

---

## 📦 전체 시스템 구조

```
동영상 분석 시스템
├── video_processing.py    # ffmpeg 기반 처리
├── stt_utils.py          # Whisper STT
├── multimodal_utils.py   # 멀티모달 LLM
└── video_analyzer.py     # 통합 분석기

입력:
├── 동영상 파일 (.mp4, .avi, .mov 등)

출력:
├── frames/               # 추출된 이미지
├── audio.wav            # 추출된 오디오
├── analysis_result.json # 종합 분석 결과
├── editing_guide.txt    # 편집 가이드
└── editing_data.json    # 편집 소프트웨어용
```

---

## 🎬 편집 소프트웨어 연동

### Adobe Premiere Pro

```python
# XML 마커 생성
def generate_premiere_markers(results):
    """Premiere Pro 마커 XML 생성"""
    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<markers>')
    
    for scene in results['scenes']:
        xml.append(f'  <marker time="{scene["timestamp"]}">')
        xml.append(f'    <name>Scene {scene["scene_number"]}</name>')
        xml.append(f'    <comment>{scene.get("description", "")[:100]}</comment>')
        xml.append('  </marker>')
    
    xml.append('</markers>')
    
    return '\n'.join(xml)
```

### DaVinci Resolve

```python
# EDL 형식 생성
def generate_edl(results):
    """EDL (Edit Decision List) 생성"""
    edl = ['TITLE: Video Analysis']
    edl.append('FCM: NON-DROP FRAME\n')
    
    for i, scene in enumerate(results['scenes'], 1):
        edl.append(f"{i:03d}  BL  V  C  {format_timecode(scene['timestamp'])}")
        edl.append(f"* FROM CLIP NAME: Scene {i}")
        edl.append(f"* COMMENT: {scene.get('description', '')[:50]}\n")
    
    return '\n'.join(edl)
```

---

## 💡 팁과 트릭

### 1. 성능 최적화

```python
# 긴 동영상은 청크로 나눠서 처리
def process_long_video(video_path, chunk_duration=300):
    """5분씩 나눠서 처리"""
    # ffmpeg로 분할
    # 각 청크 분석
    # 결과 병합
```

### 2. 배치 처리

```bash
# 여러 동영상 일괄 처리
for video in data/videos/*.mp4; do
    python src/video_analyzer.py "$video"
done
```

### 3. GPU 메모리 관리

```python
# 멀티모달 모델을 필요할 때만 로딩
analyzer = VideoAnalyzer(video_path, multimodal_model=None)
results = analyzer.analyze_full_pipeline()

# 이미지 분석이 필요한 경우만
model = MultiModalModel("llava-hf/llava-1.5-7b-hf")
for scene in results['scenes']:
    scene['description'] = model.generate_from_image(scene['frame_path'])
```

---

## 🎓 요약

### 완전한 파이프라인

```bash
# 1단계: 의존성 설치
pip install -r requirements_video.txt
pip install -r requirements_multimodal.txt
brew install ffmpeg

# 2단계: 동영상 분석
python src/video_analyzer.py video.mp4 --use_multimodal

# 3단계: 결과 확인
cat outputs/video_analysis/editing_guide.txt

# 4단계: 편집 소프트웨어에서 활용
# editing_data.json 사용
```

### 핵심 기능

✅ **ffmpeg**: 동영상 → 이미지 + 오디오
✅ **Whisper**: 음성 → 텍스트 (타임스탬프 포함)
✅ **멀티모달 LLM**: 이미지 분석
✅ **통합 분석**: 시각 + 청각 결합
✅ **편집 메타데이터**: 자동 생성

**동영상 편집이 훨씬 쉬워집니다!** 🎬✨

