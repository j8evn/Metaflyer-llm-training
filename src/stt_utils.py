"""
Speech-to-Text (STT) 유틸리티
OpenAI Whisper를 사용한 음성 인식
"""

import os
import logging
from typing import List, Dict, Optional
import whisper
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STTProcessor:
    """Speech-to-Text 처리 클래스"""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: str = "ko"
    ):
        """
        Args:
            model_size: Whisper 모델 크기 (tiny, base, small, medium, large)
            device: 디바이스 (cuda, cpu, auto)
            language: 언어 코드 (ko, en, ja 등)
        """
        self.model_size = model_size
        self.language = language
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Whisper 모델 로딩: {model_size}")
        logger.info(f"디바이스: {self.device}")
        
        self.model = whisper.load_model(model_size, device=self.device)
        
        logger.info("STT 모델 로딩 완료")
    
    def transcribe(
        self,
        audio_path: str,
        with_timestamps: bool = True
    ) -> Dict:
        """
        오디오 파일을 텍스트로 변환
        
        Args:
            audio_path: 오디오 파일 경로
            with_timestamps: 타임스탬프 포함 여부
        
        Returns:
            {
                'text': 전체 텍스트,
                'segments': [{'start': float, 'end': float, 'text': str}, ...],
                'language': 감지된 언어
            }
        """
        logger.info(f"음성 인식 시작: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        # Whisper 실행
        result = self.model.transcribe(
            audio_path,
            language=self.language if self.language != "auto" else None,
            word_timestamps=with_timestamps,
            verbose=False
        )
        
        # 결과 정리
        output = {
            'text': result['text'],
            'language': result.get('language', self.language),
            'segments': []
        }
        
        if with_timestamps and 'segments' in result:
            for seg in result['segments']:
                output['segments'].append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip()
                })
        
        logger.info(f"음성 인식 완료: {len(output['text'])} 문자")
        logger.info(f"세그먼트 수: {len(output['segments'])}")
        
        return output
    
    def get_transcript_at_time(
        self,
        segments: List[Dict],
        timestamp: float,
        context_window: float = 5.0
    ) -> str:
        """
        특정 시간의 대사 가져오기
        
        Args:
            segments: 세그먼트 리스트
            timestamp: 조회할 시간 (초)
            context_window: 전후 몇 초의 대사를 포함할지
        
        Returns:
            해당 시간대의 대사
        """
        start_time = max(0, timestamp - context_window / 2)
        end_time = timestamp + context_window / 2
        
        relevant_texts = []
        
        for seg in segments:
            if seg['start'] >= start_time and seg['end'] <= end_time:
                relevant_texts.append(seg['text'])
        
        return ' '.join(relevant_texts)


def get_whisper_model_info():
    """Whisper 모델 정보"""
    models = {
        'tiny': {'params': '39M', 'speed': '⭐⭐⭐⭐⭐', 'accuracy': '⭐⭐'},
        'base': {'params': '74M', 'speed': '⭐⭐⭐⭐', 'accuracy': '⭐⭐⭐'},
        'small': {'params': '244M', 'speed': '⭐⭐⭐', 'accuracy': '⭐⭐⭐⭐'},
        'medium': {'params': '769M', 'speed': '⭐⭐', 'accuracy': '⭐⭐⭐⭐'},
        'large': {'params': '1550M', 'speed': '⭐', 'accuracy': '⭐⭐⭐⭐⭐'}
    }
    
    print("Whisper 모델 크기:")
    print("-" * 60)
    for name, info in models.items():
        print(f"{name:8s} | {info['params']:6s} | 속도: {info['speed']} | 정확도: {info['accuracy']}")
    print("-" * 60)
    print("권장: base (빠르고 정확)")


if __name__ == "__main__":
    # 모델 정보 출력
    get_whisper_model_info()
    
    # 테스트
    audio_file = "outputs/video_analysis/audio.wav"
    
    if os.path.exists(audio_file):
        stt = STTProcessor(model_size="base")
        result = stt.transcribe(audio_file)
        
        print(f"\n전체 텍스트:\n{result['text']}")
        print(f"\n세그먼트 수: {len(result['segments'])}")
        
        # 처음 3개 세그먼트 출력
        for i, seg in enumerate(result['segments'][:3]):
            print(f"\n[{i+1}] {seg['start']:.2f}s - {seg['end']:.2f}s")
            print(f"    {seg['text']}")
    else:
        print(f"테스트 오디오 파일이 없습니다: {audio_file}")
        print("먼저 video_processing.py로 오디오를 추출하세요")

