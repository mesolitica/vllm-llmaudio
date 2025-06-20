<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## Getting Started

1. Install vLLM,

```bash
wget https://wheels.vllm.ai/b6553be1bc75f046b00046a4ad7576364d03c835/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"
pip3 install -e .
```

Or if you wish to skip isolated build environment and dependencies,

```bash
pip3 install -e . --no-build-isolation --no-deps -v
```

2. Serve,

```bash
vllm serve "mesolitica/Malaysian-Qwen2.5-7B-Audio-Instruct" --hf_overrides '{"architectures": ["LLMAudioForConditionalGeneration"]}' --trust-remote-code
```

### Using OpenAI

```python
import base64
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

with open('speech/mallm-2.mp3', 'rb') as fopen:
    audio_base64 = base64.b64encode(fopen.read()).decode('utf-8')
  
model = 'mesolitica/Malaysian-Qwen2.5-7B-Audio-Instruct'

chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": "mp3"
                },
            },
            {
                "type": "text",
                "text": "explain the audio"
            },
        ],
        
    }],
    model=model,
    max_completion_tokens=1024,
    temperature=0.6,
    top_p=0.9,
)
print(chat_completion_from_base64)
```

Output,

```
ChatCompletion(id='chatcmpl-4343d53b608249c49cc56257b7fc6c72', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The speaker is listing the consequences of shirk (polytheism) in a straightforward and matter-of-fact manner. Here’s a breakdown of each point:\n\n1. **A. Rosak akidah (Rusak iman)**: This means "ruining one\'s faith." The speaker is emphasizing that shirk directly affects one\'s spiritual well-being and faith.\n\n2. **B. Berdosa besar (Berbuat dosa besar)**: This translates to "committing major sins." The speaker is highlighting that shirk is considered a grave sin in Islam, which can lead to severe spiritual consequences.\n\n3. **C. Menjadi murtad (Menjadi muhaddid)**: This means "becoming an apostate." The speaker is pointing out that shirk can lead to a person being labeled as an apostate, which has serious legal and social implications.\n\n4. **D. Mendapat azab di akhirat (Menerima azab di akhirat)**: This translates to "receiving punishment in the afterlife." The speaker is indicating that the ultimate consequence of shirk is eternal punishment in the hereafter.\n\nThe tone is serious and educational, typical of religious teachings where the gravity of certain actions is emphasized. The speaker is likely addressing an audience that is familiar with Islamic concepts and is providing a clear and concise explanation of the consequences of shirk.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=None)], created=1750396026, model='mesolitica/Malaysian-Qwen2.5-7B-Audio-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=285, prompt_tokens=569, total_tokens=854, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None, kv_transfer_params=None)
```

### Using Requests

```python
import base64
import requests

model = 'mesolitica/Malaysian-Qwen2.5-7B-Audio-Instruct'

with open('speech/mallm-2.mp3', 'rb') as fopen:
    audio_base64 = base64.b64encode(fopen.read()).decode('utf-8')

data = {
    'messages': [{
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": "mp3"
                },
            },
            {
                "type": "text",
                "text": "explain the audio"
            },
        ],
        
    }],
    'model': model,
    'max_completion_tokens': 1024,
    'temperature': 0.6,
    'top_p': 0.9,
}

r = requests.post('http://localhost:8001/v1/chat/completions', json = data)
print(r.json())
```

Output,

```
{'id': 'chatcmpl-77279bebc2be40258f15ed90fcd813ca',
 'object': 'chat.completion',
 'created': 1750396256,
 'model': 'mesolitica/Malaysian-Qwen2.5-7B-Audio-Instruct',
 'choices': [{'index': 0,
   'message': {'role': 'assistant',
    'reasoning_content': None,
    'content': 'The speaker is reciting the consequences of committing the sin of "syirik" (shirk), which is the act of associating partners with Allah in worship or believing in other deities besides Allah. This is a fundamental concept in Islam, where shirk is considered the greatest sin.\n\nThe speaker lists the following consequences of shirk:\n\n1. **Rosa\' akidah**: This means losing one\'s faith or belief. It implies that the person\'s core beliefs are fundamentally flawed or corrupted.\n2. **Berdoa besar**: This could mean that the person\'s prayers or acts of worship are not accepted, or that they are considered significant but ultimately ineffective due to their shirk.\n3. **Menjadi murtad**: This means becoming an apostate. In Islam, apostasy is a severe sin, often leading to severe consequences, including death in some contexts.\n4. **Mendapat azab di akhirat**: This means receiving punishment in the hereafter (the afterlife). It suggests that the person will face eternal punishment for their sins, specifically their shirk.\n\nThe tone is serious and matter-of-fact, reflecting the gravity of the topic. The speaker is likely delivering this information in a formal setting, such as a sermon or educational lecture, where the importance of understanding and avoiding shirk is emphasized.\n\nThis excerpt serves as a reminder of the severe consequences of associating others with Allah and the importance of maintaining pure monotheistic beliefs in Islam.',
    'tool_calls': []},
   'logprobs': None,
   'finish_reason': 'stop',
   'stop_reason': None}],
 'usage': {'prompt_tokens': 569,
  'total_tokens': 871,
  'completion_tokens': 302,
  'prompt_tokens_details': None},
 'prompt_logprobs': None,
 'kv_transfer_params': None}
```