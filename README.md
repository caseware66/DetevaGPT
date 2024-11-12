# Introduction
As AI models evolve, detecting AI-generated text becomes increasingly challenging. Advanced models, especially when multiple LLMs collaborate in generating content, can effectively evade detection systems, mimicking human writing with high fluency. Traditional methods often fail to distinguish between AI and human-authored text. To address these challenges, we propose an enhanced detection framework that excels even against sophisticated evasion attempts. First, we leverage open source models (Gemma2-9b, Qwen2-7b, GPT-2-Large) to calculate token loss and perplexity, strengthening detection accuracy. Second, we incorporate semantic, part-of-speech (PoS), and bigram features to capture subtle human-like nuances. Our method demonstrates superior performance in identifying AI-generated content, even when crafted using Multi-LLM techniques for evasion. This work offers new insights into robust content analysis and improves the detection of AI-generated text.
<p align="center">
  <img src="https://github.com/user-attachments/assets/b88a562e-5615-4a86-a600-305204a80c46" alt="The text generation process of multi-LLM system">
  <br>
  The text generation process of multi-LLM system
</p>

# DetevaGPT
