"""
멀티모달 모델 사용 예제
"""

import sys
sys.path.append('../src')

from multimodal_utils import MultiModalModel
from PIL import Image
import requests
from io import BytesIO


def example_1_basic_inference():
    """기본 추론 예제"""
    print("=" * 60)
    print("예제 1: 기본 이미지 설명")
    print("=" * 60)
    
    # 모델 초기화
    model = MultiModalModel(
        model_name="llava-hf/llava-1.5-7b-hf",
        model_type="llava"
    )
    
    # 이미지 설명
    description = model.generate_from_image(
        image_path="data/images/test.jpg",
        prompt="이 이미지를 자세히 설명해주세요.",
        max_new_tokens=200
    )
    
    print(f"\n생성된 설명:")
    print(description)
    print()


def example_2_visual_qa():
    """Visual Q&A 예제"""
    print("=" * 60)
    print("예제 2: Visual Question Answering")
    print("=" * 60)
    
    model = MultiModalModel(
        model_name="llava-hf/llava-1.5-7b-hf",
        model_type="llava"
    )
    
    # 질문-답변
    questions = [
        "이 이미지에서 무엇을 볼 수 있나요?",
        "주요 객체의 색상은 무엇인가요?",
        "이미지의 분위기는 어떤가요?"
    ]
    
    for question in questions:
        answer = model.generate_from_image(
            image_path="data/images/test.jpg",
            prompt=f"질문: {question}\n답변:",
            max_new_tokens=100
        )
        
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    
    print()


def example_3_batch_processing():
    """배치 처리 예제"""
    print("=" * 60)
    print("예제 3: 배치 처리")
    print("=" * 60)
    
    model = MultiModalModel(
        model_name="llava-hf/llava-1.5-7b-hf",
        model_type="llava"
    )
    
    # 여러 이미지 처리
    image_paths = [
        "data/images/image1.jpg",
        "data/images/image2.jpg",
        "data/images/image3.jpg"
    ]
    
    prompts = [
        "이 이미지를 설명하세요",
        "이 이미지를 설명하세요",
        "이 이미지를 설명하세요"
    ]
    
    results = model.batch_generate(image_paths, prompts)
    
    for img_path, result in zip(image_paths, results):
        print(f"\n이미지: {img_path}")
        print(f"설명: {result}")
    
    print()


def example_4_url_image():
    """URL 이미지 사용 예제"""
    print("=" * 60)
    print("예제 4: URL 이미지 사용")
    print("=" * 60)
    
    # URL에서 이미지 다운로드
    image_url = "https://example.com/image.jpg"
    
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        # 임시 저장
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # 모델 사용
        model = MultiModalModel(
            model_name="llava-hf/llava-1.5-7b-hf",
            model_type="llava"
        )
        
        description = model.generate_from_image(
            temp_path,
            "이 이미지를 설명하세요"
        )
        
        print(f"\n설명: {description}")
        
        # 정리
        import os
        os.remove(temp_path)
    
    except Exception as e:
        print(f"오류: {e}")
    
    print()


def example_5_product_analysis():
    """제품 분석 예제"""
    print("=" * 60)
    print("예제 5: 제품 이미지 분석")
    print("=" * 60)
    
    model = MultiModalModel(
        model_name="outputs/multimodal_checkpoints/final_model",
        model_type="llava"
    )
    
    product_image = "data/images/product.jpg"
    
    # 다양한 분석
    analyses = {
        "설명": "이 제품을 설명하세요",
        "장점": "이 제품의 장점은 무엇인가요?",
        "타겟 고객": "이 제품의 타겟 고객은 누구인가요?",
        "판매 문구": "이 제품의 매력적인 판매 문구를 작성하세요"
    }
    
    for title, prompt in analyses.items():
        result = model.generate_from_image(
            product_image,
            prompt,
            max_new_tokens=150
        )
        
        print(f"\n{title}:")
        print(result)
    
    print()


if __name__ == "__main__":
    print("\n멀티모달 모델 예제")
    print("=" * 60)
    print("주의: 실제 실행하려면 이미지 파일과 학습된 모델이 필요합니다")
    print("=" * 60)
    
    # 사용 가능한 예제만 실행
    # example_1_basic_inference()
    # example_2_visual_qa()
    # example_3_batch_processing()
    # example_5_product_analysis()
    
    print("\n각 예제의 주석을 해제하여 실행하세요!")


