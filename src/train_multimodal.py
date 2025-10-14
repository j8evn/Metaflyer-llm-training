"""
멀티모달 모델 파인튜닝 스크립트
Vision-Language 모델 학습
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    Blip2ForConditionalGeneration,
    InstructBlipForConditionalGeneration
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from multimodal_utils import MultiModalDatasetLoader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """설정 파일 로딩"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_multimodal_model(model_name: str, model_type: str, config: dict):
    """멀티모달 모델 로딩"""
    logger.info(f"멀티모달 모델 로딩: {model_name} (타입: {model_type})")
    
    # Processor 로딩
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 모델 로딩
    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    if model_type == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
    elif model_type == "blip2":
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
    elif model_type == "instructblip":
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    logger.info(f"모델 로딩 완료")
    
    # 파라미터 정보
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"총 파라미터: {total_params / 1e6:.2f}M")
    
    return model, processor


def apply_lora_to_multimodal(model, config: dict):
    """멀티모달 모델에 LoRA 적용"""
    lora_config = config.get('lora', {})
    
    if not lora_config.get('use_lora', False):
        logger.info("LoRA를 사용하지 않습니다")
        return model
    
    logger.info("멀티모달 모델에 LoRA 적용 중...")
    
    # 양자화된 모델 준비
    if config.get('quantization', {}).get('use_quantization', False):
        model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정
    # Vision-Language 모델의 경우 language model 부분에만 적용
    target_modules = lora_config.get('target_modules', [
        "q_proj", "v_proj"  # 기본값
    ])
    
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        target_modules=target_modules,
        bias='none',
    )
    
    model = get_peft_model(model, peft_config)
    
    # 학습 가능한 파라미터
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    logger.info(f"LoRA 적용 완료")
    logger.info(f"학습 가능: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")
    logger.info(f"학습 비율: {100 * trainable / total:.2f}%")
    
    return model


def create_training_arguments(config: dict) -> TrainingArguments:
    """TrainingArguments 생성"""
    training_config = config['training']
    
    args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config.get('warmup_steps', 100),
        
        # 저장 설정
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        
        # 로깅
        logging_dir=f"{training_config['output_dir']}/logs",
        logging_steps=training_config.get('logging_steps', 10),
        
        # 최적화
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,  # 멀티모달에서 중요
        
        # 기타
        seed=42,
        run_name=f"multimodal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    return args


def train_multimodal(config: dict):
    """멀티모달 학습 메인 함수"""
    logger.info("=" * 50)
    logger.info("멀티모달 모델 파인튜닝 시작")
    logger.info("=" * 50)
    
    # 모델 로딩
    model_name = config['model']['name']
    model_type = config['model'].get('type', 'llava')
    
    logger.info(f"\n단계 1: 모델 및 Processor 로딩")
    model, processor = load_multimodal_model(model_name, model_type, config)
    
    # LoRA 적용
    model = apply_lora_to_multimodal(model, config)
    
    # 데이터셋 로딩
    logger.info(f"\n단계 2: 멀티모달 데이터셋 로딩")
    
    data_config = config['data']
    dataset_loader = MultiModalDatasetLoader(
        processor=processor,
        max_length=data_config.get('max_length', 512)
    )
    
    train_dataset = dataset_loader.load_from_json(data_config['train_path'])
    train_dataset = dataset_loader.prepare_dataset(train_dataset)
    
    # 검증 데이터
    eval_dataset = None
    if 'eval_path' in data_config and os.path.exists(data_config['eval_path']):
        eval_dataset = dataset_loader.load_from_json(data_config['eval_path'])
        eval_dataset = dataset_loader.prepare_dataset(eval_dataset)
    
    # Training Arguments
    logger.info(f"\n단계 3: 학습 설정")
    training_args = create_training_arguments(config)
    
    # Trainer
    logger.info(f"\n단계 4: Trainer 초기화")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 학습 시작
    logger.info(f"\n단계 5: 학습 시작")
    logger.info("=" * 50)
    
    try:
        trainer.train()
        logger.info("학습 완료!")
        
        # 모델 저장
        output_dir = config['training']['output_dir']
        final_output_dir = f"{output_dir}/final_model"
        
        logger.info(f"\n모델 저장: {final_output_dir}")
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
        
        logger.info("모델 저장 완료!")
        
    except Exception as e:
        logger.error(f"학습 중 오류: {e}", exc_info=True)
        raise
    
    logger.info("=" * 50)
    logger.info("멀티모달 파인튜닝 완료")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="멀티모달 모델 파인튜닝")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal_config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument("--model_name", type=str, help="모델 이름")
    parser.add_argument("--model_type", type=str, help="모델 타입 (llava, blip2, instructblip)")
    parser.add_argument("--dataset_path", type=str, help="데이터셋 경로")
    parser.add_argument("--output_dir", type=str, help="출력 디렉토리")
    
    args = parser.parse_args()
    
    # 설정 로딩
    if not os.path.exists(args.config):
        logger.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # 커맨드 라인 인자 병합
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.model_type:
        config['model']['type'] = args.model_type
    if args.dataset_path:
        config['data']['train_path'] = args.dataset_path
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # 출력 디렉토리 생성
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # 학습 시작
    train_multimodal(config)


if __name__ == "__main__":
    main()


