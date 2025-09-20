# LLM-chatbot
Fine-tuning PhoGPT-7.5B model in order to making LLM chatbot

# Fundamental:
Github: https://github.com/VinAIResearch/PhoGPT
Huggingface: https://huggingface.co/vinai/PhoGPT-7B5-Instruct
PhoGPT: Generative Pre-training for Vietnamese (https://arxiv.org/abs/2311.02945)
MPT model: https://huggingface.co/docs/transformers/model_doc/mpt ; https://www.mosaicml.com/mpt
Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs (https://www.databricks.com/blog/mpt-7b)
PEFT: https://huggingface.co/docs/peft/en/index ; https://arxiv.org/abs/2205.05638
LoRA: https://huggingface.co/docs/diffusers/en/training/lora ; https://arxiv.org/abs/2106.09685
Quantization: https://huggingface.co/docs/accelerate/en/usage_guides/quantization ; https://arxiv.org/abs/2208.07339
Flash attention: https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention ; https://arxiv.org/abs/2205.14135
ALiBi: Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (https://arxiv.org/abs/2108.12409)

# Fine-tuning model PhoGPT-7.5B
Model architecture: MPT-7B (https://www.databricks.com/blog/mpt-7b)
Cấu hình tối thiểu: GPU 15GB ~ Colab T4
Dataset cá nhân siêu nhỏ, tuỳ nhu cầu sử dụng
Thời gian train ~1s/item
File LoRA khá nhỏ gọn ~256MB, tổng cộng tất cả các files của 1 checkpoint khoảng dưới 800MB
Training loss xuống cũng khá nhanh, nhưng lại không bị Overfit sớm
Khi dùng Peft Fine-tuning với QLoRA-4bit thì giảm được bộ nhớ GPU rất nhiều nên chỉ cần dùng em T4-15GB colab, nhưng Inference 16bfloat thì lại phải gọi đến em V100-16GB colab
Ví dụ: nếu doanh nghiệp bạn có khoảng 1000 câu hỏi đáp, thì dataset sẽ là 1000 items, Fine-tuning khoảng 30 epoch (ChatGPT-3 fine-tuning khoảng 16 epochs ?! để overfit dữ liệu; còn GPT2 thì "We then fine-tuned a model on each of the datasets, with 500 training epochs per model 😂, a threshold chosen to prevent overfitting."_page4 of https://arxiv.org/pdf/1908.09203.pdf) để cho em nó overfit thì 1000 items x 30 epochs = 30.000 iterations / 3600s = 8.33h Nvidia Tesla T4 colab x $0.42/h = $3.58

# Config
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained(
mode_id,
quantization_config=bnb_config,
device_map="cua",
trust_remote_code=True)
alt text

peft_config = LoraConfig(
r=32,
lora_alpha=32,
target_modules=["attn.Wqkv", "attn.out_proj", "ffn.up_proj", "ffn.down_proj"],
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM")
training_arguments = TrainingArguments(
output_dir=output_model,
per_device_train_batch_size=1,
gradient_accumulation_steps=1,
optim="paged_adamw_32bit",
learning_rate=1e-4,
lr_scheduler_type="cosine",
save_strategy="steps",
save_steps=50,
logging_steps=10,
num_train_epochs=100,
max_steps=100,
fp16=True)
trainer = SFTTrainer(
model=model,
train_dataset=data,
peft_config=peft_config,
dataset_text_field="text",
args=training_arguments,
tokenizer=tokenizer,
packing=False,
max_seq_length=1024)
Model Size: 3.7B
Context length: 8192
Vocab size: 20K
Training data size: 70K instructional prompt and response pairs & 290K conversations
Note: PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"
