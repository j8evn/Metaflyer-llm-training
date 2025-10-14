"""
멀티모달 모델 유틸리티
이미지-텍스트 Vision-Language 모델 지원
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Blip2ForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalDatasetLoader:
    """멀티모달 데이터셋 로딩 및 전처리 클래스"""
    
    def __init__(
        self,
        processor,  # AutoProcessor
        max_length: int = 512
    ):
        self.processor = processor
        self.max_length = max_length
    
    def load_from_json(self, file_path: str) -> Dataset:
        """
        JSON 파일에서 멀티모달 데이터셋 로딩
        
        형식:
        [
            {
                "image": "path/to/image.jpg",
                "text": "이미지 설명",
                "question": "질문 (선택)",
                "answer": "답변"
            }
        ]
        """
        logger.info(f"멀티모달 데이터 로딩: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON 데이터는 리스트 형식이어야 합니다")
        
        # 데이터 검증
        for i, item in enumerate(data):
            if 'image' not in item:
                raise ValueError(f"샘플 {i}에 'image' 키가 없습니다")
        
        logger.info(f"로딩된 샘플 수: {len(data)}")
        return Dataset.from_list(data)
    
    def process_batch(self, examples: Dict) -> Dict:
        """
        배치 데이터 전처리
        """
        images = []
        texts = []
        
        for i in range(len(examples['image'])):
            # 이미지 로딩
            image_path = examples['image'][i]
            
            # 절대 경로가 아니면 상대 경로로 처리
            if not os.path.isabs(image_path):
                # data 디렉토리 기준으로 찾기
                base_dirs = ['data/images', 'data', '.']
                for base_dir in base_dirs:
                    full_path = os.path.join(base_dir, image_path)
                    if os.path.exists(full_path):
                        image_path = full_path
                        break
            
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            except Exception as e:
                logger.error(f"이미지 로딩 실패: {image_path} - {e}")
                # 빈 이미지 생성
                images.append(Image.new('RGB', (224, 224), color='white'))
            
            # 텍스트 구성
            if 'question' in examples and examples['question'][i]:
                # Q&A 형식
                text = f"Question: {examples['question'][i]}\nAnswer: {examples['answer'][i]}"
            elif 'text' in examples and examples['text'][i]:
                # 캡션 형식
                text = examples['text'][i]
            else:
                # answer만 있는 경우
                text = examples.get('answer', [''])[i]
            
            texts.append(text)
        
        # Processor로 인코딩
        encodings = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Labels 설정 (input_ids와 동일)
        encodings['labels'] = encodings['input_ids'].clone()
        
        return encodings
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """데이터셋 전처리"""
        logger.info("멀티모달 데이터셋 전처리 시작")
        
        # 기존 컬럼 저장
        original_columns = dataset.column_names
        
        # 전처리 적용
        processed_dataset = dataset.map(
            self.process_batch,
            batched=True,
            batch_size=8,
            remove_columns=original_columns
        )
        
        logger.info(f"전처리 완료. 샘플 수: {len(processed_dataset)}")
        return processed_dataset


class MultiModalModel:
    """멀티모달 모델 래퍼"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "llava",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"멀티모달 모델 로딩: {model_name} (타입: {model_type})")
        
        self._load_model()
    
    def _load_model(self):
        """모델 로딩"""
        if self.model_type == "llava":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
        elif self.model_type == "blip2":
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
        elif self.model_type == "instructblip":
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        self.model.eval()
        logger.info(f"모델 로딩 완료")
    
    def generate_from_image(
        self,
        image_path: str,
        prompt: str = "이 이미지를 설명해주세요.",
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        이미지와 텍스트로부터 생성
        
        Args:
            image_path: 이미지 파일 경로
            prompt: 텍스트 프롬프트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
        """
        # 이미지 로딩
        image = Image.open(image_path).convert('RGB')
        
        # 입력 처리
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )
        
        # 디코딩
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def batch_generate(
        self,
        image_paths: List[str],
        prompts: List[str],
        max_new_tokens: int = 256
    ) -> List[str]:
        """배치 생성"""
        results = []
        
        for image_path, prompt in zip(image_paths, prompts):
            result = self.generate_from_image(
                image_path,
                prompt,
                max_new_tokens
            )
            results.append(result)
        
        return results


def create_sample_multimodal_dataset(
    output_path: str,
    num_samples: int = 20
):
    """
    샘플 멀티모달 데이터셋 생성
    
    실제로는 이미지 파일이 있어야 하지만,
    여기서는 구조만 생성합니다.
    """
    sample_data = [
        {
            "image": "data/images/cat.jpg",
            "question": "이 이미지에 무엇이 있나요?",
            "answer": "고양이가 소파에 앉아 있습니다."
        },
        {
            "image": "data/images/dog.jpg",
            "question": "동물의 품종은 무엇인가요?",
            "answer": "골든 리트리버 강아지입니다."
        },
        {
            "image": "data/images/food.jpg",
            "text": "맛있어 보이는 피자가 테이블 위에 놓여 있습니다."
        },
        {
            "image": "data/images/landscape.jpg",
            "question": "이 장소는 어디인가요?",
            "answer": "아름다운 산과 호수가 있는 자연 풍경입니다."
        },
        {
            "image": "data/images/city.jpg",
            "text": "현대적인 도시의 스카이라인과 높은 건물들이 보입니다."
        }
    ]
    
    # 샘플 반복
    full_data = []
    for i in range(num_samples):
        sample = sample_data[i % len(sample_data)].copy()
        full_data.append(sample)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"샘플 멀티모달 데이터셋 생성: {output_path}")
    logger.info(f"주의: 실제 이미지 파일을 data/images/ 디렉토리에 준비하세요")


if __name__ == "__main__":
    # 샘플 데이터셋 생성
    create_sample_multimodal_dataset("../data/multimodal_train.json", num_samples=20)
    print("샘플 멀티모달 데이터셋이 생성되었습니다!")
    print("주의: 실제 이미지 파일을 data/images/ 디렉토리에 추가하세요")


