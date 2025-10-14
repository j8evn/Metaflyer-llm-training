"""
동영상 처리 유틸리티
ffmpeg를 사용한 장면 추출 및 오디오 분리
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """동영상 처리 클래스"""
    
    def __init__(self, video_path: str, output_dir: str = "outputs/video_analysis"):
        """
        Args:
            video_path: 동영상 파일 경로
            output_dir: 출력 디렉토리
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, "frames")
        self.audio_path = os.path.join(output_dir, "audio.wav")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"동영상 파일을 찾을 수 없습니다: {video_path}")
        
        logger.info(f"VideoProcessor 초기화: {video_path}")
    
    def get_video_info(self) -> Dict:
        """동영상 정보 가져오기"""
        cap = cv2.VideoCapture(self.video_path)
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        
        logger.info(f"동영상 정보:")
        logger.info(f"  - 해상도: {info['width']}x{info['height']}")
        logger.info(f"  - FPS: {info['fps']:.2f}")
        logger.info(f"  - 프레임 수: {info['frame_count']}")
        logger.info(f"  - 길이: {info['duration']:.2f}초")
        
        return info
    
    def extract_frames_by_interval(
        self,
        interval_seconds: float = 1.0,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        일정 간격으로 프레임 추출
        
        Args:
            interval_seconds: 추출 간격 (초)
            max_frames: 최대 프레임 수 (None = 제한 없음)
        
        Returns:
            추출된 이미지 파일 경로 리스트
        """
        logger.info(f"프레임 추출 시작 (간격: {interval_seconds}초)")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="프레임 추출")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 지정된 간격마다 저장
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frame_filename = f"frame_{saved_count:06d}_t{timestamp:.2f}s.jpg"
                frame_path = os.path.join(self.frames_dir, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
                
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        logger.info(f"프레임 추출 완료: {saved_count}개")
        return frame_paths
    
    def extract_frames_by_scene(
        self,
        threshold: float = 30.0,
        min_scene_duration: float = 1.0
    ) -> List[str]:
        """
        장면 변화 감지하여 프레임 추출
        
        Args:
            threshold: 장면 변화 감지 임계값 (높을수록 덜 민감)
            min_scene_duration: 최소 장면 길이 (초)
        
        Returns:
            추출된 이미지 파일 경로 리스트
        """
        logger.info(f"장면 변화 감지 프레임 추출 시작")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_scene_frames = int(fps * min_scene_duration)
        
        frame_paths = []
        prev_frame = None
        frame_count = 0
        saved_count = 0
        last_scene_frame = -min_scene_frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="장면 분석")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # 프레임 차이 계산
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = diff.mean()
                
                # 장면 변화 감지
                if mean_diff > threshold and (frame_count - last_scene_frame) > min_scene_frames:
                    timestamp = frame_count / fps
                    frame_filename = f"scene_{saved_count:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(self.frames_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_count += 1
                    last_scene_frame = frame_count
            
            prev_frame = gray
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        logger.info(f"장면 추출 완료: {saved_count}개 장면")
        return frame_paths
    
    def extract_audio(self) -> str:
        """
        오디오 추출
        
        Returns:
            추출된 오디오 파일 경로
        """
        logger.info("오디오 추출 시작")
        
        # ffmpeg 사용
        cmd = [
            'ffmpeg',
            '-i', self.video_path,
            '-vn',  # 비디오 스트림 제외
            '-acodec', 'pcm_s16le',  # WAV 포맷
            '-ar', '16000',  # 16kHz (Whisper 권장)
            '-ac', '1',  # 모노
            '-y',  # 덮어쓰기
            self.audio_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"오디오 추출 완료: {self.audio_path}")
            return self.audio_path
        
        except subprocess.CalledProcessError as e:
            logger.error(f"오디오 추출 실패: {e.stderr.decode()}")
            raise
    
    def get_frame_timestamp(self, frame_filename: str) -> float:
        """프레임 파일명에서 타임스탬프 추출"""
        # frame_000001_t5.23s.jpg → 5.23
        import re
        match = re.search(r't([\d.]+)s', frame_filename)
        if match:
            return float(match.group(1))
        return 0.0


class SceneDetector:
    """장면 감지 클래스 (고급)"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
    
    def detect_scenes(
        self,
        threshold: float = 30.0,
        min_scene_length: int = 15
    ) -> List[Dict]:
        """
        장면 감지 및 분석
        
        Returns:
            장면 정보 리스트 [{'start_frame': int, 'end_frame': int, 'duration': float}, ...]
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        scenes = []
        prev_frame = None
        scene_start = 0
        frame_count = 0
        
        logger.info("장면 감지 중...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # 마지막 장면 추가
                if frame_count - scene_start > min_scene_length:
                    scenes.append({
                        'start_frame': scene_start,
                        'end_frame': frame_count,
                        'start_time': scene_start / fps,
                        'end_time': frame_count / fps,
                        'duration': (frame_count - scene_start) / fps
                    })
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray).mean()
                
                if diff > threshold and (frame_count - scene_start) > min_scene_length:
                    # 새 장면 시작
                    scenes.append({
                        'start_frame': scene_start,
                        'end_frame': frame_count,
                        'start_time': scene_start / fps,
                        'end_time': frame_count / fps,
                        'duration': (frame_count - scene_start) / fps
                    })
                    scene_start = frame_count
            
            prev_frame = gray
            frame_count += 1
        
        cap.release()
        
        logger.info(f"장면 감지 완료: {len(scenes)}개 장면")
        return scenes


def check_ffmpeg_installed() -> bool:
    """ffmpeg 설치 확인"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


if __name__ == "__main__":
    # 사용 예제
    if not check_ffmpeg_installed():
        print("오류: ffmpeg가 설치되지 않았습니다.")
        print("설치: brew install ffmpeg  (Mac)")
        print("      sudo apt install ffmpeg  (Ubuntu)")
        exit(1)
    
    # 예제 실행
    video_file = "test_video.mp4"
    
    if os.path.exists(video_file):
        processor = VideoProcessor(video_file)
        
        # 정보 출력
        info = processor.get_video_info()
        
        # 프레임 추출
        frames = processor.extract_frames_by_interval(interval_seconds=2.0)
        print(f"추출된 프레임: {len(frames)}개")
        
        # 오디오 추출
        audio = processor.extract_audio()
        print(f"추출된 오디오: {audio}")
    else:
        print(f"테스트 동영상 파일이 없습니다: {video_file}")

