resource "aws_s3_bucket" "aws_terraform_state_bucket" {
  bucket = "${var.org_prefix}-terraform-state"
}
