docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --entrypoint python3 \
  vllm/vllm-openai-rocm:latest \
  -c "import torch; print('HIP available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('ROCm version:', torch.version.hip)"
