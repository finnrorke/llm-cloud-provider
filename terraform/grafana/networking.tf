data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "grafana" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(local.common_tags, {
    Name = "${local.stack_name}-vpc"
  })
}

resource "aws_internet_gateway" "grafana" {
  vpc_id = aws_vpc.grafana.id

  tags = merge(local.common_tags, {
    Name = "${local.stack_name}-igw"
  })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.grafana.id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = coalesce(var.availability_zone, data.aws_availability_zones.available.names[0])
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.stack_name}-public-subnet"
  })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.grafana.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.grafana.id
  }

  tags = merge(local.common_tags, {
    Name = "${local.stack_name}-public-rt"
  })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "grafana" {
  name_prefix = "${local.stack_name}-"
  description = "Allow Grafana access and optional SSH access."
  vpc_id      = aws_vpc.grafana.id

  dynamic "ingress" {
    for_each = var.allowed_grafana_cidr_blocks
    content {
      description = "Grafana"
      from_port   = var.grafana_port
      to_port     = var.grafana_port
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  dynamic "ingress" {
    for_each = var.allowed_ssh_cidr_blocks
    content {
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  egress {
    description = "Outbound internet access"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.stack_name}-sg"
  })
}
