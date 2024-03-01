import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer


def as_chat(data):
    message = f"""
### HUMAN:
請將引號內的華語改寫，結果的第 1 行為漢字，第 2 行為羅馬字
"{data['華語']}"

### RESPONSE:
"{data['漢字']}"
"{data['羅馬字']}"
    """
    return dict(chat=message.strip())


def main():
    train_dataset = load_from_disk('dataset/train')
    # converted = train_dataset.map(as_chat)
    # for batch in converted:
    #     print(batch['chat'])

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
    # data = data.map(lambda data_point: tokenizer(
    #         transform_data(
    #             data_point["conversations"],
    #             tokenizer.eos_token,
    #             instruct=instruct
    #             ),
    #         max_length=context_window,
    #         truncation=True,
    #     ))


if __name__ == '__main__':
    main()
