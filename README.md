# llm-cloud-provider
This is a deployable, multi-cloud LLM inference provider

export AWS_PROFILE=xxx
cd terraform/envs/dev
Setup backend.hcl file and tfvars
run teraform init -backend-config=backend.hcl
terraform plan -var-file=xxx
terraform apply -var-file=xxx