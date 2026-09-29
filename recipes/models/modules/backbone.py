from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_backbone(name, use_lora, lora_r, lora_alpha):
    if "SmolLM" in name:
        model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(name)
    elif "Qwen3" in name:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(name)
    else:
        raise NotImplementedError(f"Backbone model {name} is not supported.")

    if use_lora:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return tokenizer, model
