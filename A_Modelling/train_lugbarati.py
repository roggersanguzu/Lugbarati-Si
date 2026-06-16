import os
import torch
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

def load_model_safely(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    for param in model.parameters():
        if param.device.type == "meta":
            param.data = torch.empty(param.shape, dtype=torch.float32, device="cpu")
    model = model.to("cpu")
    return model

def main():
    tokenized_data_path = "./data/lugbarati_tokenized_build"
    base_model_architecture = "facebook/nllb-200-distilled-600M"
    output_model_dir = "./data/sunbird_lugbarati_model"
    final_save_path = "./data/lugbarati_sunbird_final"

    if not os.path.exists(tokenized_data_path):
        print(f"Cannot find tokenized data at {tokenized_data_path}")
        return

    tokenized_datasets = load_from_disk(tokenized_data_path)

    print("Loading model...")
    model = load_model_safely(base_model_architecture)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_architecture,
        src_lang="lgb_Latn"
    )

    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if param.device.type == "meta":
            param.data = torch.empty(param.shape, dtype=torch.float32, device="cpu")

    model = model.to("cpu")
    model.train()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=False,
        use_cpu=True,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        save_strategy="epoch",
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        max_length=128,
        pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Model saved at: {final_save_path}")

if __name__ == "__main__":
    main()