"""
동영상 종합 분석기
이미지(시각) + 텍스트(청각) 통합 분석
"""

import os
import json
import logging
from typing import List, Dict, Optional
from datetime import timedelta

from video_processing import VideoProcessor, SceneDetector
from stt_utils import STTProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """동영상 종합 분석 클래스"""
    
    def __init__(
        self,
        video_path: str,
        multimodal_model=None,  # MultiModalModel 인스턴스
        output_dir: str = "outputs/video_analysis"
    ):
        """
        Args:
            video_path: 동영상 파일 경로
            multimodal_model: 멀티모달 모델 (선택)
            output_dir: 출력 디렉토리
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.multimodal_model = multimodal_model
        
        # 비디오 프로세서
        self.video_processor = VideoProcessor(video_path, output_dir)
        
        # STT 프로세서
        self.stt_processor = STTProcessor(model_size="base")
        
        logger.info(f"VideoAnalyzer 초기화: {video_path}")
    
    def analyze_full_pipeline(
        self,
        extract_method: str = "interval",  # "interval" or "scene"
        interval_seconds: float = 2.0,
        scene_threshold: float = 30.0
    ) -> Dict:
        """
        전체 분석 파이프라인 실행
        
        Returns:
            {
                'video_info': {...},
                'frames': [...],
                'transcript': {...},
                'scenes': [...],
                'metadata': {...}
            }
        """
        logger.info("=" * 60)
        logger.info("동영상 종합 분석 시작")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. 동영상 정보
        logger.info("\n[1/4] 동영상 정보 수집")
        video_info = self.video_processor.get_video_info()
        results['video_info'] = video_info
        
        # 2. 프레임 추출
        logger.info(f"\n[2/4] 프레임 추출 (방식: {extract_method})")
        
        if extract_method == "interval":
            frame_paths = self.video_processor.extract_frames_by_interval(
                interval_seconds=interval_seconds
            )
        else:
            frame_paths = self.video_processor.extract_frames_by_scene(
                threshold=scene_threshold
            )
        
        results['frames'] = frame_paths
        
        # 3. 오디오 추출 및 STT
        logger.info("\n[3/4] 음성 인식 (STT)")
        audio_path = self.video_processor.extract_audio()
        transcript = self.stt_processor.transcribe(audio_path)
        results['transcript'] = transcript
        
        # 4. 장면 분석 (이미지 + 대사 통합)
        logger.info("\n[4/4] 장면 종합 분석")
        scenes = self._analyze_scenes(frame_paths, transcript['segments'])
        results['scenes'] = scenes
        
        # 5. 메타데이터 생성
        metadata = self._generate_metadata(results)
        results['metadata'] = metadata
        
        # 결과 저장
        output_file = os.path.join(self.output_dir, "analysis_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n분석 결과 저장: {output_file}")
        logger.info("=" * 60)
        logger.info("동영상 분석 완료!")
        logger.info("=" * 60)
        
        return results
    
    def _analyze_scenes(
        self,
        frame_paths: List[str],
        transcript_segments: List[Dict]
    ) -> List[Dict]:
        """
        장면별 종합 분석
        
        Args:
            frame_paths: 프레임 이미지 경로 리스트
            transcript_segments: 대사 세그먼트
        
        Returns:
            장면 분석 결과 리스트
        """
        scenes = []
        
        for i, frame_path in enumerate(frame_paths):
            # 타임스탬프 추출
            timestamp = self.video_processor.get_frame_timestamp(
                os.path.basename(frame_path)
            )
            
            # 해당 시간의 대사 찾기
            dialogue = self.stt_processor.get_transcript_at_time(
                transcript_segments,
                timestamp,
                context_window=3.0
            )
            
            scene = {
                'scene_number': i + 1,
                'timestamp': timestamp,
                'frame_path': frame_path,
                'dialogue': dialogue,
            }
            
            # 멀티모달 모델로 이미지 분석 (선택)
            if self.multimodal_model:
                try:
                    # 기본 설명
                    description = self.multimodal_model.generate_from_image(
                        frame_path,
                        "이 장면을 설명해주세요.",
                        max_new_tokens=150
                    )
                    scene['description'] = description
                    
                    # 감정/분위기 분석
                    mood = self.multimodal_model.generate_from_image(
                        frame_path,
                        "이 장면의 분위기나 감정을 설명해주세요.",
                        max_new_tokens=50
                    )
                    scene['mood'] = mood
                    
                    # 주요 객체
                    objects = self.multimodal_model.generate_from_image(
                        frame_path,
                        "이 장면의 주요 객체나 인물을 나열해주세요.",
                        max_new_tokens=100
                    )
                    scene['objects'] = objects
                    
                except Exception as e:
                    logger.warning(f"이미지 분석 실패 (장면 {i+1}): {e}")
                    scene['description'] = ""
                    scene['mood'] = ""
                    scene['objects'] = ""
            
            scenes.append(scene)
            
            # 진행 상황 출력
            if (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{len(frame_paths)} 장면 분석 완료")
        
        return scenes
    
    def _generate_metadata(self, results: Dict) -> Dict:
        """
        편집용 메타데이터 생성
        
        Returns:
            동영상 편집에 유용한 메타데이터
        """
        metadata = {
            'video_file': os.path.basename(self.video_path),
            'duration': results['video_info']['duration'],
            'total_scenes': len(results['scenes']),
            'has_audio': bool(results['transcript']['text']),
            'analysis_summary': {}
        }
        
        # 장면별 요약
        scene_types = []
        dialogues = []
        
        for scene in results['scenes']:
            if scene.get('mood'):
                scene_types.append(scene['mood'])
            if scene.get('dialogue'):
                dialogues.append(scene['dialogue'])
        
        metadata['analysis_summary'] = {
            'total_dialogue_segments': len(results['transcript']['segments']),
            'scenes_with_dialogue': sum(1 for s in results['scenes'] if s.get('dialogue')),
            'scenes_analyzed': len([s for s in results['scenes'] if s.get('description')])
        }
        
        return metadata
    
    def generate_editing_guide(self, results: Dict) -> str:
        """
        동영상 편집 가이드 생성
        
        Returns:
            편집 가이드 텍스트
        """
        guide = []
        guide.append("=" * 60)
        guide.append("동영상 편집 가이드")
        guide.append("=" * 60)
        guide.append("")
        
        guide.append(f"동영상: {os.path.basename(self.video_path)}")
        guide.append(f"길이: {results['video_info']['duration']:.2f}초")
        guide.append(f"총 장면: {len(results['scenes'])}개")
        guide.append("")
        
        guide.append("=" * 60)
        guide.append("장면별 분석")
        guide.append("=" * 60)
        
        for scene in results['scenes']:
            guide.append(f"\n[장면 {scene['scene_number']}] - {scene['timestamp']:.2f}초")
            guide.append(f"  프레임: {os.path.basename(scene['frame_path'])}")
            
            if scene.get('description'):
                guide.append(f"  설명: {scene['description'][:100]}...")
            
            if scene.get('mood'):
                guide.append(f"  분위기: {scene['mood']}")
            
            if scene.get('dialogue'):
                guide.append(f"  대사: {scene['dialogue'][:100]}...")
            
            if scene.get('objects'):
                guide.append(f"  객체: {scene['objects'][:80]}...")
        
        guide.append("\n" + "=" * 60)
        guide.append("편집 제안")
        guide.append("=" * 60)
        
        # 자동 편집 제안 생성
        suggestions = self._generate_editing_suggestions(results)
        for suggestion in suggestions:
            guide.append(f"  • {suggestion}")
        
        return '\n'.join(guide)
    
    def _generate_editing_suggestions(self, results: Dict) -> List[str]:
        """편집 제안 생성"""
        suggestions = []
        
        scenes = results['scenes']
        
        # 대사 없는 긴 장면 찾기
        silent_scenes = [
            s for s in scenes
            if not s.get('dialogue') and s.get('timestamp', 0) > 0
        ]
        
        if len(silent_scenes) > len(scenes) * 0.3:
            suggestions.append("대사 없는 장면이 많습니다. BGM 추가를 고려하세요.")
        
        # 장면 전환 빈도
        if len(scenes) > results['video_info']['duration'] / 2:
            suggestions.append("장면 전환이 빠릅니다. 일부 장면을 결합하는 것을 고려하세요.")
        
        # 감정 변화 분석
        if any(s.get('mood') for s in scenes):
            suggestions.append("감정/분위기 변화가 있습니다. 적절한 전환 효과를 사용하세요.")
        
        return suggestions
    
    def export_for_editing(
        self,
        results: Dict,
        format: str = "premiere"
    ) -> str:
        """
        편집 소프트웨어용 데이터 export
        
        Args:
            results: 분석 결과
            format: 'premiere', 'davinci', 'json'
        
        Returns:
            Export 파일 경로
        """
        if format == "json":
            # JSON 형식
            output_file = os.path.join(self.output_dir, "editing_data.json")
            
            editing_data = {
                'video': os.path.basename(self.video_path),
                'markers': [],
                'scenes': []
            }
            
            for scene in results['scenes']:
                editing_data['markers'].append({
                    'time': scene['timestamp'],
                    'label': f"Scene {scene['scene_number']}",
                    'description': scene.get('description', '')[:100]
                })
                
                editing_data['scenes'].append({
                    'start': scene['timestamp'],
                    'description': scene.get('description', ''),
                    'dialogue': scene.get('dialogue', ''),
                    'mood': scene.get('mood', '')
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(editing_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"편집 데이터 export 완료: {output_file}")
            return output_file
        
        else:
            logger.warning(f"지원하지 않는 형식: {format}")
            return ""


def format_time(seconds: float) -> str:
    """초를 HH:MM:SS 형식으로 변환"""
    return str(timedelta(seconds=int(seconds)))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="동영상 종합 분석")
    parser.add_argument("video_path", type=str, help="동영상 파일 경로")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/video_analysis",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="프레임 추출 간격 (초)"
    )
    parser.add_argument(
        "--use_multimodal",
        action="store_true",
        help="멀티모달 모델 사용 (이미지 분석)"
    )
    parser.add_argument(
        "--multimodal_model_path",
        type=str,
        help="멀티모달 모델 경로"
    )
    
    args = parser.parse_args()
    
    # 멀티모달 모델 로딩 (선택)
    multimodal_model = None
    if args.use_multimodal and args.multimodal_model_path:
        try:
            from multimodal_utils import MultiModalModel
            logger.info("멀티모달 모델 로딩 중...")
            multimodal_model = MultiModalModel(
                model_name=args.multimodal_model_path,
                model_type="llava"
            )
        except Exception as e:
            logger.warning(f"멀티모달 모델 로딩 실패: {e}")
    
    # 분석기 초기화
    analyzer = VideoAnalyzer(
        args.video_path,
        multimodal_model=multimodal_model,
        output_dir=args.output_dir
    )
    
    # 전체 분석 실행
    results = analyzer.analyze_full_pipeline(
        extract_method="interval",
        interval_seconds=args.interval
    )
    
    # 편집 가이드 생성
    guide = analyzer.generate_editing_guide(results)
    
    guide_file = os.path.join(args.output_dir, "editing_guide.txt")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("\n" + guide)
    print(f"\n편집 가이드 저장: {guide_file}")
    
    # 편집용 데이터 export
    export_file = analyzer.export_for_editing(results, format="json")
    print(f"편집 데이터 export: {export_file}")

