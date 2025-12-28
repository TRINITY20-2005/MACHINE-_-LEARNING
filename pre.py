from transformers import GPT2Config, GPT2LMHeadModel

# Initializing a configuration (Defining the architecture)
config = GPT2Config(
    vocab_size=50257,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Initializing the model with random weights (No pre-trained data)
model = GPT2LMHeadModel(config)

print(f"Model initialized with {model.num_parameters()} parameters.")
# Output: Model initialized with 124439808 parameters.