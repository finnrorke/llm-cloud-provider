docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ROCM_USE_AITER=0 \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3-0.6B \
  --dtype auto \
  --enforce-eager \
  --max-model-len 4096
