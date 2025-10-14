#!/bin/bash

# 동영상 분석 스크립트
# 사용법: ./scripts/analyze_video.sh video.mp4 [options]

set -e

# 색상
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}동영상 분석 파이프라인${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# 인자 확인
if [ $# -lt 1 ]; then
    echo "사용법: ./scripts/analyze_video.sh <video_file> [options]"
    echo ""
    echo "옵션:"
    echo "  --interval SECONDS    프레임 추출 간격 (기본: 2.0)"
    echo "  --multimodal         멀티모달 분석 사용"
    echo "  --model PATH         멀티모달 모델 경로"
    echo ""
    echo "예제:"
    echo "  ./scripts/analyze_video.sh video.mp4"
    echo "  ./scripts/analyze_video.sh video.mp4 --interval 1.0"
    echo "  ./scripts/analyze_video.sh video.mp4 --multimodal --model llava-hf/llava-1.5-7b-hf"
    exit 1
fi

VIDEO_FILE=$1
shift

# ffmpeg 설치 확인
echo -e "${YELLOW}[1/4] 환경 확인...${NC}"
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg가 설치되지 않았습니다."
    echo "설치 방법:"
    echo "  Mac: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    exit 1
fi
echo -e "${GREEN}✓ ffmpeg 설치됨${NC}"

# 동영상 파일 확인
if [ ! -f "$VIDEO_FILE" ]; then
    echo "동영상 파일을 찾을 수 없습니다: $VIDEO_FILE"
    exit 1
fi
echo -e "${GREEN}✓ 동영상 파일: $VIDEO_FILE${NC}"
echo ""

# 분석 실행
echo -e "${YELLOW}[2/4] 동영상 분석 시작...${NC}"
python src/video_analyzer.py "$VIDEO_FILE" "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[3/4] 분석 완료!${NC}"
    echo ""
    
    # 결과 확인
    echo -e "${YELLOW}[4/4] 결과 확인${NC}"
    OUTPUT_DIR="outputs/video_analysis"
    
    echo "출력 파일:"
    echo "  📊 분석 결과: $OUTPUT_DIR/analysis_result.json"
    echo "  📝 편집 가이드: $OUTPUT_DIR/editing_guide.txt"
    echo "  🎬 편집 데이터: $OUTPUT_DIR/editing_data.json"
    echo "  🖼️  프레임: $OUTPUT_DIR/frames/"
    echo "  🔊 오디오: $OUTPUT_DIR/audio.wav"
    echo ""
    
    # 편집 가이드 미리보기
    if [ -f "$OUTPUT_DIR/editing_guide.txt" ]; then
        echo "편집 가이드 미리보기:"
        echo "---"
        head -n 20 "$OUTPUT_DIR/editing_guide.txt"
        echo "..."
        echo "---"
        echo ""
        echo "전체 가이드: cat $OUTPUT_DIR/editing_guide.txt"
    fi
    
    echo ""
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}분석 완료! 🎉${NC}"
    echo -e "${GREEN}=====================================${NC}"
else
    echo ""
    echo "분석 실패. 로그를 확인하세요."
    exit 1
fi

