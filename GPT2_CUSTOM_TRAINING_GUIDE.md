# GPT-2 Custom Training Guide for AI Honeypot Simulator

This guide explains how to customize and fine-tune the GPT-2 model for your specific honeypot requirements.

## Configuration Parameters

The following parameters can be configured via environment variables or by modifying `config.py`:

### Model Selection
- `GPT2_MODEL_NAME`: Choose the GPT-2 model size
  - `"gpt2"` (124M parameters) - Default, fastest
  - `"gpt2-medium"` (355M parameters) - Better quality
  - `"gpt2-large"` (774M parameters) - High quality, slower
  - `"gpt2-xl"` (1.5B parameters) - Best quality, requires significant GPU memory

### Generation Parameters
- `GPT2_MAX_NEW_TOKENS`: Maximum tokens to generate (default: 150)
- `GPT2_TEMPERATURE`: Controls randomness (0.1-2.0, default: 0.8)
  - Lower = more deterministic
  - Higher = more creative/random
- `GPT2_TOP_P`: Nucleus sampling parameter (0.1-1.0, default: 0.9)
- `GPT2_TOP_K`: Top-K sampling parameter (1-100, default: 50)
- `GPT2_DO_SAMPLE`: Enable sampling (true/false, default: true)

### Example Environment Configuration
```bash
# Use larger model for better quality
export GPT2_MODEL_NAME="gpt2-medium"

# More conservative generation
export GPT2_TEMPERATURE="0.6"
export GPT2_TOP_P="0.8"
export GPT2_MAX_NEW_TOKENS="200"

# For deterministic output (testing)
export GPT2_DO_SAMPLE="false"
export GPT2_TEMPERATURE="0.1"
```

## Custom Training Options

### Option 1: Fine-tuning with Custom Data

To fine-tune GPT-2 on your own honeypot-specific data:

1. **Prepare Training Data**
   ```python
   # Create training data file: honeypot_training_data.txt
   # Format: One example per line, separated by <|endoftext|>
   
   # Example content:
   """
   # Apache Configuration
   ServerRoot /etc/apache2
   Listen 80
   LoadModule rewrite_module modules/mod_rewrite.so
   <|endoftext|>
   
   [INFO] User login successful from 192.168.1.100
   [WARN] High CPU usage detected: 85%
   [ERROR] Database connection failed
   <|endoftext|>
   
   admin:x:1000:1000:Admin User:/home/admin:/bin/bash
   guest:x:1001:1001:Guest Account:/home/guest:/bin/bash
   <|endoftext|>
   """
   ```

2. **Fine-tuning Script**
   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
   from transformers import TextDataset, DataCollatorForLanguageModeling
   
   def fine_tune_gpt2(train_file, model_name="gpt2", output_dir="./fine_tuned_gpt2"):
       # Load model and tokenizer
       tokenizer = GPT2Tokenizer.from_pretrained(model_name)
       model = GPT2LMHeadModel.from_pretrained(model_name)
       
       # Add padding token
       tokenizer.pad_token = tokenizer.eos_token
       
       # Prepare dataset
       dataset = TextDataset(
           tokenizer=tokenizer,
           file_path=train_file,
           block_size=128
       )
       
       data_collator = DataCollatorForLanguageModeling(
           tokenizer=tokenizer,
           mlm=False
       )
       
       # Training arguments
       training_args = TrainingArguments(
           output_dir=output_dir,
           overwrite_output_dir=True,
           num_train_epochs=3,
           per_device_train_batch_size=4,
           save_steps=1000,
           save_total_limit=2,
           prediction_loss_only=True,
           logging_steps=100,
           warmup_steps=100,
           learning_rate=5e-5,
       )
       
       # Create trainer
       trainer = Trainer(
           model=model,
           args=training_args,
           data_collator=data_collator,
           train_dataset=dataset,
       )
       
       # Train
       trainer.train()
       trainer.save_model()
       tokenizer.save_pretrained(output_dir)
   
   # Usage
   fine_tune_gpt2("honeypot_training_data.txt")
   ```

3. **Use Fine-tuned Model**
   ```bash
   export GPT2_MODEL_NAME="./fine_tuned_gpt2"
   ```

### Option 2: Domain-Specific Prompting

Enhance generation without training by improving prompts:

1. **Create Custom Templates**
   ```python
   # In src/ai_engine/text_generator.py, modify templates:
   
   CUSTOM_TEMPLATES = {
       "security_logs": [
           "[SECURITY] Intrusion attempt detected from {ip} targeting {service}",
           "[SECURITY] Failed authentication for user {user} from {ip}",
           "[SECURITY] Suspicious file access: {file} by {user}",
       ],
       "network_configs": [
           "# Network Security Configuration\nfirewall_enabled=true\nallow_ssh_from={ip_range}",
           "# VPN Configuration\nvpn_server={server}\nencryption=AES-256",
       ]
   }
   ```

2. **Context-Aware Generation**
   ```python
   def generate_with_context(self, content_type, context_vars):
       template = random.choice(CUSTOM_TEMPLATES.get(content_type, ["Generic content"]))
       prompt = template.format(**context_vars)
       return self.generator(prompt, **self.generation_params)
   ```

### Option 3: Multi-Model Ensemble

Combine multiple models for different content types:

```python
class MultiModelGenerator:
    def __init__(self):
        self.models = {
            "logs": GPT2LMHeadModel.from_pretrained("./log_specialized_gpt2"),
            "configs": GPT2LMHeadModel.from_pretrained("./config_specialized_gpt2"),
            "general": GPT2LMHeadModel.from_pretrained("gpt2")
        }
    
    def generate(self, content_type, prompt):
        model = self.models.get(content_type, self.models["general"])
        # Generate using appropriate model
```

## Performance Optimization

### GPU Memory Management
```python
# In config.py, add:
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "true").lower() == "true"

# For low-memory GPUs:
export GPU_MEMORY_FRACTION="0.5"
export ENABLE_GRADIENT_CHECKPOINTING="true"
export GPT2_MODEL_NAME="gpt2"  # Use smaller model
```

### Batch Processing
```python
# Process multiple requests together
def generate_batch(self, prompts, **kwargs):
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = self.model.generate(**inputs, **kwargs)
    return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
```

## Quality Improvement Tips

1. **Temperature Tuning**
   - Start with 0.8, adjust based on output quality
   - Lower for more consistent output
   - Higher for more creative/varied output

2. **Prompt Engineering**
   - Include specific context in prompts
   - Use consistent formatting
   - Provide examples in prompts

3. **Content Filtering**
   - Implement post-processing filters
   - Remove unwanted patterns
   - Ensure realistic content

4. **Evaluation Metrics**
   - Monitor generation time
   - Check content relevance
   - Measure attacker engagement

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use smaller model
2. **Slow Generation**: Enable GPU, reduce max_tokens
3. **Poor Quality**: Increase model size, improve prompts
4. **Repetitive Output**: Adjust temperature and top_p

### Debug Mode
```bash
export GPT2_DEBUG="true"
export GPT2_LOG_GENERATIONS="true"
```

This will log all generations for analysis and improvement.