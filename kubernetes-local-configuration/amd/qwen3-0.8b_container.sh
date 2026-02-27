docker run --rm \
  --device=/dev/kfd \ # Add the amd fusion driver
  --device=/dev/dri \ # expose rendering nodes through the direct rendering infrastructure driver
  --group-add video \ # Needs video and render to access the GPU device files
  --group-add render \
  --ipc=host \ # Shares hosts memory namespace
  --cap-add=SYS_PTRACE \ # ROCM debugging and profiling tool
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ROCM_USE_AITER=0 \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3-0.6B \
  --dtype auto \
  --enforce-eager \
  --max-model-len 4096
