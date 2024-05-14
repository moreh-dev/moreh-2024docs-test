---
icon: terminal
tags: [component]
order: 100
expanded: true
---

Fine-tuning은 미리 학습된 기계 학습 모델을 새로운 데이터나 특정 작업에 재조정하는 행위를 의미합니다. 즉, AI 모델을 새로운 작업에 적용하고자 할 때, 기존 모델에 새로운 데이터 셋을 학습 시켜 최적화 해나가는 과정인데요. Fine-tuning을 통해 기존 모델을 가져와서 사용자의 필요 및 특정 도메인에 맞게 특화시킬 수 있습니다.

일반적으로 사전학습된 모델을 미세조정(Fine-tuning) 하는데 사전학습된 모델은 범용성을 고려한 매우 큰 파라미터를 가지는 모델이며 큰 모델을 효과적으로 fine-tuning하려면 수백 개에서 수천 개의 예제가 필요합니다.

MoAI Platform에서는 GPU의 메모리 사이즈를 고려해 최적화된 병렬화 기법을 손쉽게 적용할 수 있어 학습 시작 전에 소요되는 시간과 노력을 획기적으로 줄일 수 있습니다.

이 튜토리얼에서는 다음 5종의 모델을 fine-tuning 과정을 배우고, MoAI Platform 을 사용하여 직접 손쉽게 구현하는 방법에 대해 알아보겠습니다.



## Fine-tuning Tutorials

- [Llama2](/Tutorials/Llama2_Tutorial/index.md)
- [Mistral](/Tutorials/Mistral_Tutorial/index.md)
- [GPT](/Tutorials/GPT_Tutorial/index.md)
- [Qwen](/Tutorials/Qwen_Tutorial/index.md)
- [Baichuan2](/Tutorials/Baichuan2_Tutorial/index.md)


### Fine-Tuning 방법

모델을 Fine-tuning 과정은 다음과 같습니다:

1. 사전 학습된 모델과 작업에 특화된 데이터셋을 준비합니다.
2. 데이터셋의 예제를 모델에 전달하고 모델의 출력을 수집합니다.
3. 모델의 출력과 예상 출력 간의 손실을 계산합니다.
4. 경사 하강법과 역전파를 사용하여 손실을 줄이기 위해 모델 파라미터를 업데이트합니다.
5. 모델이 수렴될 때까지 여러 epoch에 대해 단계 3-5를 반복합니다.
6. Fine-tuned 모델은 이제 새로운 데이터에 대한 추론을 위해 배포될 수 있는 상태가 됩니다.