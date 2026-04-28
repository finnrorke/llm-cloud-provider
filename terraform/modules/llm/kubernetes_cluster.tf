data "aws_ssm_parameter" "ubuntu_2204_ami" {
  name = "/aws/service/canonical/ubuntu/server/jammy/stable/current/amd64/hvm/ebs-gp2/ami-id"
}

data "aws_ssm_parameter" "gpu_test_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

data "aws_subnet" "control_plane" {
  id = var.control_plane_subnet_id
}

data "aws_subnet" "cpu_infra" {
  id = var.cpu_infra_subnet_id
}

resource "aws_instance" "control_plane" {
  ami                    = data.aws_ssm_parameter.ubuntu_2204_ami.value
  instance_type          = var.control_plane_instance_type
  subnet_id              = var.control_plane_subnet_id
  vpc_security_group_ids = [var.kubernetes_control_plane_security_group_id]
  user_data = templatefile("${path.module}/templates/k3s-control-plane.sh.tftpl", {
    cluster_token = var.cluster_token
    cluster_cidr  = var.cluster_cidr
    service_cidr  = var.service_cidr
  })
  user_data_replace_on_change = true

  metadata_options {
    http_tokens = "required"
  }

  tags = merge(
    var.common_tags,
    {
      Name = "${var.stack_name}-control-plane"
      Role = "kubernetes-control-plane"
    },
  )

  lifecycle {
    precondition {
      condition     = data.aws_subnet.control_plane.vpc_id == var.vpc_id
      error_message = "The control plane subnet must belong to the provided VPC."
    }

    precondition {
      condition     = data.aws_subnet.control_plane.availability_zone == var.availability_zone
      error_message = "The control plane subnet must be in the configured availability zone."
    }
  }
}

resource "aws_instance" "cpu_infra" {
  ami                    = data.aws_ssm_parameter.ubuntu_2204_ami.value
  instance_type          = var.cpu_infra_instance_type
  subnet_id              = var.cpu_infra_subnet_id
  vpc_security_group_ids = [var.kubernetes_node_security_group_id]
  user_data = templatefile("${path.module}/templates/k3s-worker.sh.tftpl", {
    cluster_token      = var.cluster_token
    control_plane_host = aws_instance.control_plane.private_ip
    node_pool          = "cpu-infra"
    node_label_extra   = ""
  })
  user_data_replace_on_change = true

  metadata_options {
    http_tokens = "required"
  }

  tags = merge(
    var.common_tags,
    {
      Name = "${var.stack_name}-cpu-infra"
      Role = "kubernetes-node"
    },
  )

  lifecycle {
    precondition {
      condition     = data.aws_subnet.cpu_infra.vpc_id == var.vpc_id
      error_message = "The CPU infra subnet must belong to the provided VPC."
    }

    precondition {
      condition     = data.aws_subnet.cpu_infra.availability_zone == var.availability_zone
      error_message = "The CPU infra subnet must be in the configured availability zone."
    }
  }
}

resource "aws_instance" "gpu_test" {
  count                  = var.gpu_test_instance_count
  ami                    = data.aws_ssm_parameter.gpu_test_ami.value
  instance_type          = var.gpu_test_instance_type
  subnet_id              = var.cpu_infra_subnet_id
  vpc_security_group_ids = [var.kubernetes_node_security_group_id]
  user_data = templatefile("${path.module}/templates/k3s-worker.sh.tftpl", {
    cluster_token      = var.cluster_token
    control_plane_host = aws_instance.control_plane.private_ip
    node_pool          = "gpu-test"
    node_label_extra   = "accelerator=nvidia"
  })
  user_data_replace_on_change = true

  metadata_options {
    http_tokens = "required"
  }

  tags = merge(
    var.common_tags,
    {
      Name     = "${var.stack_name}-gpu-test-${count.index + 1}"
      Role     = "kubernetes-node"
      NodePool = "gpu-test"
    },
  )

  lifecycle {
    precondition {
      condition     = data.aws_subnet.cpu_infra.vpc_id == var.vpc_id
      error_message = "The GPU test subnet must belong to the provided VPC."
    }

    precondition {
      condition     = data.aws_subnet.cpu_infra.availability_zone == var.availability_zone
      error_message = "The GPU test subnet must be in the configured availability zone."
    }
  }
}
