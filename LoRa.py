from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Configure LoRA
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["c_attn"], 
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 82,207,488 || trainable%: 0.3587