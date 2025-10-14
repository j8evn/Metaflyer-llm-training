"""
멀티모달 모델 추론 스크립트
이미지-텍스트 Vision-Language 모델 추론
"""

import os
import sys
import argparse
import logging
from PIL import Image

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Blip2ForConditionalGeneration,
    InstructBlipForConditionalGeneration
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalInference:
    """멀티모달 추론 엔진"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "llava",
        device: str = "auto"
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"멀티모달 추론 엔진 초기화")
        logger.info(f"모델: {model_path}")
        logger.info(f"타입: {model_type}")
        logger.info(f"디바이스: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """모델 로딩"""
        logger.info("모델 로딩 중...")
        
        # Processor 로딩
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # 모델 로딩
        model_kwargs = {
            'pretrained_model_name_or_path': self.model_path,
            'device_map': 'auto' if self.device == 'cuda' else None,
            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32
        }
        
        if self.model_type == "llava":
            self.model = LlavaForConditionalGeneration.from_pretrained(**model_kwargs)
        elif self.model_type == "blip2":
            self.model = Blip2ForConditionalGeneration.from_pretrained(**model_kwargs)
        elif self.model_type == "instructblip":
            self.model = InstructBlipForConditionalGeneration.from_pretrained(**model_kwargs)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        self.model.eval()
        logger.info("모델 로딩 완료")
    
    def generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        이미지-텍스트 생성
        
        Args:
            image_path: 이미지 파일 경로
            prompt: 텍스트 프롬프트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_p: Top-p 샘플링
        """
        # 이미지 로딩
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
        
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
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # 디코딩
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def describe_image(self, image_path: str, max_new_tokens: int = 256) -> str:
        """이미지 설명 생성"""
        prompt = "이 이미지를 자세히 설명해주세요."
        return self.generate(image_path, prompt, max_new_tokens)
    
    def answer_question(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256
    ) -> str:
        """이미지에 대한 질문 답변"""
        prompt = f"질문: {question}\n답변:"
        return self.generate(image_path, prompt, max_new_tokens)


def interactive_mode(engine: MultiModalInference):
    """대화형 모드"""
    logger.info("\n" + "=" * 50)
    logger.info("멀티모달 대화형 모드")
    logger.info("=" * 50)
    print("\n명령어:")
    print("  describe <image_path> - 이미지 설명")
    print("  ask <image_path> <question> - 이미지에 대한 질문")
    print("  quit - 종료")
    print()
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                logger.info("종료합니다.")
                break
            
            parts = command.split(maxsplit=2)
            
            if len(parts) < 2:
                print("올바른 형식: describe <image_path> 또는 ask <image_path> <question>")
                continue
            
            cmd = parts[0].lower()
            image_path = parts[1]
            
            if not os.path.exists(image_path):
                print(f"이미지를 찾을 수 없습니다: {image_path}")
                continue
            
            print("\n생성 중...")
            
            if cmd == "describe":
                result = engine.describe_image(image_path)
                print(f"\n설명:\n{result}\n")
            
            elif cmd == "ask":
                if len(parts) < 3:
                    print("질문을 입력하세요: ask <image_path> <question>")
                    continue
                
                question = parts[2]
                result = engine.answer_question(image_path, question)
                print(f"\n답변:\n{result}\n")
            
            else:
                print("알 수 없는 명령어. describe 또는 ask를 사용하세요.")
        
        except KeyboardInterrupt:
            logger.info("\n\n종료합니다.")
            break
        except Exception as e:
            logger.error(f"오류: {e}")


def main():
    parser = argparse.ArgumentParser(description="멀티모달 모델 추론")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="학습된 멀티모달 모델 경로"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="llava",
        choices=["llava", "blip2", "instructblip"],
        help="모델 타입"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="이미지 파일 경로"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="텍스트 프롬프트"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="이미지에 대한 질문"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="생성할 최대 토큰 수"
    )
    
    args = parser.parse_args()
    
    # 추론 엔진 초기화
    engine = MultiModalInference(
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # 추론 모드
    if args.image and args.prompt:
        # 단일 생성
        logger.info(f"이미지: {args.image}")
        logger.info(f"프롬프트: {args.prompt}")
        
        result = engine.generate(
            image_path=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens
        )
        
        print(f"\n생성 결과:\n{result}\n")
    
    elif args.image and args.question:
        # Q&A
        logger.info(f"이미지: {args.image}")
        logger.info(f"질문: {args.question}")
        
        result = engine.answer_question(
            image_path=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens
        )
        
        print(f"\n답변:\n{result}\n")
    
    elif args.image:
        # 이미지 설명
        logger.info(f"이미지: {args.image}")
        
        result = engine.describe_image(
            image_path=args.image,
            max_new_tokens=args.max_new_tokens
        )
        
        print(f"\n설명:\n{result}\n")
    
    else:
        # 대화형 모드
        interactive_mode(engine)


if __name__ == "__main__":
    main()


