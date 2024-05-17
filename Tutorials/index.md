---
icon: terminal
tags: tags: [tutorial]
order: 100
expanded: true
---

This guide is for anyone who wants to fine-tune powerful language models such as Llama2, Mistral, and etc for their own projects.
We will walk through the steps to finetune these large language models (LLMs) with MoAI Platform.

## Fine-tuning Tutorials

- [Llama2](/Tutorials/Llama2_Tutorial/index.md)
- [Mistral](/Tutorials/Mistral_Tutorial/index.md)
- [GPT](/Tutorials/GPT_Tutorial/index.md)
- [Qwen](/Tutorials/Qwen_Tutorial/index.md)
- [Baichuan2](/Tutorials/Baichuan2_Tutorial/index.md)


Fine-tuning in machine learning involves adjusting a pre-trained machine learning model's weight on new data to enhance task-specific performance. Essentially, when you want to apply an AI model to a new task, you take an existing model and optimize it with new datasets. This allows you to customize the model to meet your specific needs and domain requirements.

Typically, fine-tuning a pre-trained model involves a model with a large number of parameters designed for general-purpose use, and effectively fine-tuning such a large model requires hundreds to thousands of examples.

With the MoAI Platform, you can easily apply optimized parallelization techniques that consider the GPU's memory size, significantly reducing the time and effort needed before starting training.


What you will learn here:

1. How to find and prep datasets
2. Turning datasets into ChatML format for training
3. Choosing the right training settings
