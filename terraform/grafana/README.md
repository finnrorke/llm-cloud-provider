# Grafana EC2 stack

This Terraform stack creates a small AWS environment for a single Grafana host:

- a dedicated VPC
- one public subnet with an internet gateway
- a security group for Grafana and optional SSH
- one `t3.micro` EC2 instance running Grafana OSS

## Usage

```sh
cd terraform/grafana
cp variables.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Restrict `allowed_grafana_cidr_blocks` and `allowed_ssh_cidr_blocks` before applying. By default the backend is local state in this directory. If you want shared remote state, replace the block in `backend.tf` with your S3 backend configuration.
