---
icon: terminal
tags: [guide]
order: 40
---

# 4. Checking Training Results

Running the **`train_mistral.py`** script, as in the previous section, will save the resulting model in the **`mistral_code_generation`** directory. This is a pure PyTorch model parameter file and is fully compatible with regular GPU servers, not just the MoAI Platform.

You can test the trained model using the **`inference_mistral.py`** script located in the **`tutorial`** directory of the GitHub repository you downloaded earlier. In this test, the prompt "Create a function that takes a list of strings as input and joins them with spaces" was used.

```python
# tutorial/inference_mistral.py
...
input_text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a function to join given list of strings with space.

### Input:\n['I', 'love', 'you']

### Output:
"""
```

코드를 실행합니다.

```python
~/quickstart$ python tutorial/inference_mistral.py
```

출력값을 확인해보면 모델이 프롬프트 내용대로 적절한 함수를 생성한 것을 확인할 수 있습니다.

```bash
Mistral: Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a function to join given list of strings with space.

### Input:
['I', 'love', 'Moreh']

### Output:
def join_strings(string_list):
		return ' '.join(string_list)

result = join_strings(['I', 'love', 'Moreh'])
print(result)
```
