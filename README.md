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
vllm serve "mesolitica/Malaysian-Qwen2.5-7B-Audio-Instruct" --hf_overrides '{"architectures": ["LLMAudioForConditionalGeneration"]}' --dtype float16
```