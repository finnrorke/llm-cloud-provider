variable "region" {
  type        = string
  description = "AWS region for the Grafana instance."
  default     = "eu-west-2"
}

variable "org_prefix" {
  type        = string
  description = "Short prefix used in AWS resource names."
}

variable "environment" {
  type        = string
  description = "Environment name used for tagging and naming."
  default     = "dev"
}

variable "name" {
  type        = string
  description = "Service name for this stack."
  default     = "grafana"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the Grafana VPC."
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidr" {
  type        = string
  description = "CIDR block for the public subnet that hosts the instance."
  default     = "10.42.1.0/24"
}

variable "availability_zone" {
  type        = string
  description = "Optional AZ override. Defaults to the first available AZ in the region."
  default     = null
  nullable    = true
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type for Grafana."
  default     = "t3.micro"
}

variable "key_name" {
  type        = string
  description = "Optional EC2 key pair for SSH access."
  default     = null
  nullable    = true
}

variable "grafana_port" {
  type        = number
  description = "Port exposed by Grafana."
  default     = 3000
}

variable "grafana_admin_user" {
  type        = string
  description = "Grafana admin username."
  default     = "admin"
}

variable "grafana_admin_password" {
  type        = string
  description = "Grafana admin password."
  sensitive   = true

  validation {
    condition     = length(var.grafana_admin_password) >= 12
    error_message = "grafana_admin_password must be at least 12 characters long."
  }
}

variable "allowed_grafana_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to reach Grafana."
  default     = ["0.0.0.0/0"]
}

variable "allowed_ssh_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to SSH to the instance. Leave empty to disable SSH ingress."
  default     = []
}

variable "root_volume_size_gb" {
  type        = number
  description = "Size of the EC2 root volume in GB."
  default     = 16
}

variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to all resources."
  default     = {}
}
