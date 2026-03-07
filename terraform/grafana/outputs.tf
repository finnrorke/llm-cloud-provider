output "instance_id" {
  description = "EC2 instance ID for the Grafana host."
  value       = aws_instance.grafana.id
}

output "public_ip" {
  description = "Public IP address of the Grafana instance."
  value       = aws_instance.grafana.public_ip
}

output "grafana_url" {
  description = "URL for the Grafana UI."
  value       = "http://${aws_instance.grafana.public_ip}:${var.grafana_port}"
}

output "vpc_id" {
  description = "VPC ID created for the Grafana stack."
  value       = aws_vpc.grafana.id
}

output "public_subnet_id" {
  description = "Public subnet ID created for the Grafana stack."
  value       = aws_subnet.public.id
}
