# Usage

### Ollama

```bash
ollama run erosynthesis/Llama-3-8B-Chinese-Chat-Jimmer-q8_0-v1:latest
# or
ollama run erosynthesis/Llama-3-8B-Chinese-Chat-Jimmer-q5_k_m-v1:latest
```

### llama.cpp

```bash
llama-cli --hf-repo EroSynthesis/Llama-3-8B-Chinese-Chat-Jimmer-Lora-q8_0-v1-GGUF --hf-file Llama-3-8B-Chinese-Chat-Jimmer-Lora-q8_0-v1.gguf -p "介绍Jimmer"
# or
llama-cli --hf-repo EroSynthesis/Llama-3-8B-Chinese-Chat-Jimmer-Lora-q5_k_m-v1-GGUF --hf-file Llama-3-8B-Chinese-Chat-Jimmer-Lora-q5_k_m-v1.gguf -p "介绍Jimmer"
```

