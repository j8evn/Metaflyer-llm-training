import os
import json
import torch
import logging
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DATA_PATH = os.path.join(BASE_DIR, "data/dataset.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "models/weights")
HF_CACHE_DIR = "/dataset/cep/cache/huggingface/hub"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. 데이터셋 (검증된 메시지 기반 구조)
class PersonDataset(torch.utils.data.Dataset):
    def __init__(self, processor):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": item["text_input"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["text_output"]}],
            }
        ]
        return messages

# 3. 데이터 콜레이터 (assistant 답변 구간 마스킹)
def data_collator(features, processor):
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in features]
    image_inputs, video_inputs = process_vision_info(features)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    labels = inputs["input_ids"].clone()
    # <|im_start|>assistant 토큰을 찾아 그 이후만 학습하도록 설정
    try:
        assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    except:
        assistant_token_id = None
    
    if assistant_token_id is not None:
        for i in range(labels.shape[0]):
            input_ids = labels[i].tolist()
            try:
                sep_idx = input_ids.index(assistant_token_id)
                # <|im_start|>assistant\n 다음인 2개 토큰 이후부터 학습 대상 설정
                labels[i, :sep_idx+2] = -100 
            except ValueError:
                pass
    
    # 패딩 토큰 마스킹
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
            
    inputs["labels"] = labels
    return inputs

# 4. 학습 메인 루틴
def train():
    logger.info(f"모델 로드 시작: {MODEL_ID}")
    torch.cuda.empty_cache()

    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 메모리 파편화 방지 설정
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # VRAM 여유 분량 확보 (70GiB 제한)
    max_memory = {i: "70GiB" for i in range(torch.cuda.device_count())}

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=HF_CACHE_DIR,
        attn_implementation="sdpa" # flash_attn 설치 불필요
    )

    # [중요] OOM 방지: prepare_model_for_kbit_training 대신 수동 설정 사용
    # 이 방식이 30B 이상 거대 모델에서 float32 캐스팅 오버헤드를 줄여줍니다.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # 가중치 고정 (LoRA만 학습)
    for param in model.parameters():
        param.requires_grad = False

    # LoRA 설정
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 이미지 해상도 제한 (256x256) - 학습 속도 및 메모리 확보
    train_pixels = 256 * 256 
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        cache_dir=HF_CACHE_DIR,
        min_pixels=28*28,
        max_pixels=train_pixels
    )

    # 학습 조건 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,   # OOM 방지를 위해 1로 하향
        gradient_accumulation_steps=16, # Effective Batch = 16 유지
        learning_rate=2e-4,
        num_train_epochs=5,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=PersonDataset(processor),
        data_collator=lambda x: data_collator(x, processor),
    )

    logger.info("학습 실행 중...")
    trainer.train()

    # 최종 어댑터 저장
    save_path = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    logger.info(f"학습 완료! 모델이 저장되었습니다: {save_path}")

if __name__ == "__main__":
    train()
