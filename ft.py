import torch
import transformers
from datasets import load_dataset, load_from_disk
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer)


def as_chat(data):
    message = f"""
### HUMAN:
請將引號內的華語改寫，結果的第 1 行為漢字，第 2 行為羅馬字
"{data['華語']}"

### RESPONSE:
"{data['漢字']}"
"{data['羅馬字']}"
    """
    return message.strip()


def main():
    model_name = "open_llama_7b"
    base_model = "open_llama_7b"
    base_model = "openlm-research/open_llama_7b"
    target_modules = ["q_proj", "k_proj", "v_proj"]
    trainer_output_dir = "fine-tune-results/"
    model_output_dir = "models/"
    instruct = False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(base_model,
                                             quantization_config=bnb_config,
                                             device_map={"": 0})

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    # Load dataset and prepare data
    context_window = 2048

    data = load_from_disk('dataset/train')
    data = data.map(lambda data_point: tokenizer(
        as_chat(
            data_point
        ),
        max_length=context_window,
        truncation=True,
    ))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs=1,  # full run
            learning_rate=2e-4,
            fp16=True,
            logging_steps=20,
            output_dir=trainer_output_dir,
            report_to="tensorboard",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()

    model_save_path = f"{model_output_dir}/{model_name}_adapter"
    trainer.save_model(model_save_path)

    base_model = LlamaForCausalLM.from_pretrained(base_model, device_map="cpu")
    model = PeftModel.from_pretrained(base_model, model_save_path)
    merged_model = model.merge_and_unload()
    model_save_path = f"model_output_dir/{model_name}"
    tokenizer.save_pretrained(model_save_path)
    merged_model.save_pretrained(model_save_path)


if __name__ == "__main__":
    main()
