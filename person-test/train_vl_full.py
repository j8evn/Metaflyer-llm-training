import json
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoProcessor
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType

# 1. 모델 설정 (H100용 고성능 설정)
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

from transformers import AutoModelForVision2Seq, AutoProcessor

# 모델 로드 (bf16 사용)
print(f"Loading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto", 
    trust_remote_code=True
)

# 2. LoRA 설정 (성능 중심)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,              # Rank 대폭 상향
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Dataset 정의
class ImageTextDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        text_input = item["text_input"]
        text_output = item["text_output"]

        # 이미지 처리
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        # 텍스트 처리
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_input},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 입력 처리 (Prompt 길이를 알기 위함)
        inputs_prompt = processor(
            text=[text],
            images=[image],
            max_pixels=512*512,
            return_tensors="pt",
        )
        
        # 전체 데이터 처리 (질문 + 답변)
        full_text = text + text_output + processor.tokenizer.eos_token
        inputs_full = processor(
            text=[full_text],
            images=[image],
            max_pixels=512*512,
            padding=False,      # Collator에서 처리
            truncation=False,   # Truncation 금지 (이미지 토큰 유지)
            return_tensors="pt",
        )
        
        input_ids = inputs_full.input_ids.squeeze(0)
        attention_mask = inputs_full.attention_mask.squeeze(0)
        labels = input_ids.clone()
        
        # Prompt 부분 마스킹
        prompt_length = inputs_prompt.input_ids.shape[1]
        labels[:prompt_length] = -100
        
        # 패딩은 Collator에서 수행되므로 여기서는 -100 마스킹 불필요 (Collator에서 처리 예정)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs_full.pixel_values,
            "image_grid_thw": inputs_full.image_grid_thw if "image_grid_thw" in inputs_full else None
        }

train_dataset = ImageTextDataset("data/dataset.json")

# 4. Collate Function
def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    
    pixel_values = [b["pixel_values"] for b in batch if b["pixel_values"] is not None]
    pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None

    image_grid_thw = [b["image_grid_thw"] for b in batch if b["image_grid_thw"] is not None]
    image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }

# 5. Training Arguments (H100 최적화)
training_args = TrainingArguments(
    output_dir="./output_full",
    overwrite_output_dir=True, # 기존 학습 결과 덮어쓰기
    per_device_train_batch_size=4,  # 배치 사이즈 증가
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,    # 메모리 여유가 있어도 켜두는 게 안전 (속도 차이 크지 않음)
    num_train_epochs=5,             # Epoch 증가
    learning_rate=2e-5,             # LoRA Rank가 높으므로 LR 조정
    bf16=True,                      # H100 필수 설정
    optim="adamw_torch",            # 표준 AdamW 사용 (Adafactor보다 성능 좋음)
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False,    # MPS 경고 방지용이었으나 Linux에서는 True여도 무방 (여기선 안전하게 False 유지)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

print("Starting High-Performance Training on H100...")
trainer.train()
