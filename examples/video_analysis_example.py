"""
동영상 분석 실전 예제
"""

import os
import sys
import json
sys.path.append('../src')

from video_processing import VideoProcessor
from stt_utils import STTProcessor
from multimodal_utils import MultiModalModel


def example_1_basic_extraction():
    """기본 추출 예제"""
    print("=" * 60)
    print("예제 1: 기본 프레임/오디오 추출")
    print("=" * 60)
    
    video_file = "data/videos/test.mp4"
    
    if not os.path.exists(video_file):
        print(f"테스트 동영상이 없습니다: {video_file}")
        return
    
    # VideoProcessor 초기화
    processor = VideoProcessor(video_file, "outputs/example1")
    
    # 동영상 정보
    info = processor.get_video_info()
    print(f"\n동영상 정보:")
    print(f"  길이: {info['duration']:.2f}초")
    print(f"  FPS: {info['fps']:.2f}")
    
    # 프레임 추출 (2초마다)
    print("\n프레임 추출 중...")
    frames = processor.extract_frames_by_interval(interval_seconds=2.0)
    print(f"추출된 프레임: {len(frames)}개")
    
    # 오디오 추출
    print("\n오디오 추출 중...")
    audio = processor.extract_audio()
    print(f"오디오 파일: {audio}")
    print()


def example_2_stt_analysis():
    """STT 분석 예제"""
    print("=" * 60)
    print("예제 2: 음성 인식 (STT)")
    print("=" * 60)
    
    audio_file = "outputs/example1/audio.wav"
    
    if not os.path.exists(audio_file):
        print(f"오디오 파일이 없습니다: {audio_file}")
        print("먼저 예제 1을 실행하세요")
        return
    
    # STT 초기화
    print("\nWhisper 모델 로딩...")
    stt = STTProcessor(model_size="base", language="ko")
    
    # 음성 인식
    print("음성 인식 중...")
    result = stt.transcribe(audio_file)
    
    print(f"\n전체 텍스트:\n{result['text']}\n")
    
    print(f"세그먼트 ({len(result['segments'])}개):")
    for i, seg in enumerate(result['segments'][:5], 1):
        print(f"  [{i}] {seg['start']:.2f}s - {seg['end']:.2f}s")
        print(f"      {seg['text']}")
    
    if len(result['segments']) > 5:
        print(f"  ... 외 {len(result['segments']) - 5}개")
    print()


def example_3_multimodal_analysis():
    """멀티모달 분석 예제"""
    print("=" * 60)
    print("예제 3: 멀티모달 이미지 분석")
    print("=" * 60)
    
    frame_dir = "outputs/example1/frames"
    
    if not os.path.exists(frame_dir):
        print(f"프레임 디렉토리가 없습니다: {frame_dir}")
        return
    
    # 멀티모달 모델 로딩
    print("\n멀티모달 모델 로딩...")
    print("(이 과정은 시간이 걸릴 수 있습니다...)")
    
    try:
        model = MultiModalModel(
            model_name="llava-hf/llava-1.5-7b-hf",
            model_type="llava"
        )
        
        # 첫 3개 프레임 분석
        frames = sorted(os.listdir(frame_dir))[:3]
        
        for frame_name in frames:
            frame_path = os.path.join(frame_dir, frame_name)
            
            print(f"\n프레임: {frame_name}")
            
            # 장면 설명
            description = model.generate_from_image(
                frame_path,
                "이 장면을 설명하세요",
                max_new_tokens=100
            )
            print(f"  설명: {description}")
            
            # 분위기
            mood = model.generate_from_image(
                frame_path,
                "이 장면의 분위기는?",
                max_new_tokens=50
            )
            print(f"  분위기: {mood}")
    
    except Exception as e:
        print(f"오류: {e}")
        print("멀티모달 모델이 설치되지 않았을 수 있습니다:")
        print("  pip install -r requirements_multimodal.txt")
    print()


def example_4_full_pipeline():
    """전체 파이프라인 예제"""
    print("=" * 60)
    print("예제 4: 전체 분석 파이프라인")
    print("=" * 60)
    
    video_file = "data/videos/test.mp4"
    
    if not os.path.exists(video_file):
        print(f"동영상 파일이 없습니다: {video_file}")
        return
    
    try:
        from video_analyzer import VideoAnalyzer
        
        # 분석기 초기화 (멀티모달 제외)
        analyzer = VideoAnalyzer(
            video_path=video_file,
            output_dir="outputs/full_analysis"
        )
        
        # 전체 분석
        print("\n전체 분석 실행 중...")
        results = analyzer.analyze_full_pipeline(
            extract_method="interval",
            interval_seconds=2.0
        )
        
        # 요약 출력
        print("\n분석 요약:")
        print(f"  총 프레임: {len(results['frames'])}개")
        print(f"  총 대사: {len(results['transcript']['segments'])}개 세그먼트")
        print(f"  분석된 장면: {len(results['scenes'])}개")
        
        # 첫 3개 장면 출력
        print("\n첫 3개 장면:")
        for scene in results['scenes'][:3]:
            print(f"\n  장면 {scene['scene_number']} ({scene['timestamp']:.2f}s)")
            print(f"    대사: {scene.get('dialogue', '')[:60]}...")
        
        # 편집 가이드
        guide = analyzer.generate_editing_guide(results)
        print("\n" + guide[:500] + "...\n")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    print()


def example_5_youtube_chapters():
    """YouTube 챕터 자동 생성"""
    print("=" * 60)
    print("예제 5: YouTube 챕터 자동 생성")
    print("=" * 60)
    
    analysis_file = "outputs/full_analysis/analysis_result.json"
    
    if not os.path.exists(analysis_file):
        print(f"분석 결과가 없습니다: {analysis_file}")
        print("먼저 예제 4를 실행하세요")
        return
    
    # 분석 결과 로딩
    with open(analysis_file, 'r') as f:
        results = json.load(f)
    
    # YouTube 챕터 생성
    print("\nYouTube 챕터 (설명란에 복사):")
    print("-" * 60)
    
    for i, scene in enumerate(results['scenes']):
        timestamp = scene['timestamp']
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        # 대사나 설명에서 주제 추출
        if scene.get('dialogue'):
            topic = scene['dialogue'][:30]
        elif scene.get('description'):
            topic = scene['description'][:30]
        else:
            topic = f"장면 {i+1}"
        
        print(f"{minutes:02d}:{seconds:02d} {topic}")
        
        if i >= 9:  # 처음 10개만
            print("...")
            break
    
    print("-" * 60)
    print()


if __name__ == "__main__":
    print("\n동영상 분석 예제 모음")
    print("=" * 60)
    
    # 실행할 예제 선택
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        if example_num == "1":
            example_1_basic_extraction()
        elif example_num == "2":
            example_2_stt_analysis()
        elif example_num == "3":
            example_3_multimodal_analysis()
        elif example_num == "4":
            example_4_full_pipeline()
        elif example_num == "5":
            example_5_youtube_chapters()
        else:
            print(f"사용법: python video_analysis_example.py [1-5]")
    else:
        print("사용법: python video_analysis_example.py [번호]")
        print("\n예제:")
        print("  1 - 기본 추출 (ffmpeg)")
        print("  2 - STT 분석 (Whisper)")
        print("  3 - 멀티모달 분석 (LLaVA)")
        print("  4 - 전체 파이프라인")
        print("  5 - YouTube 챕터 생성")
        print("\n예: python video_analysis_example.py 1")

