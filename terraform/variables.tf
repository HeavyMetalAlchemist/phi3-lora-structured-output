variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "s3_bucket_name" {
  description = "S3 bucket name for training artifacts"
  type        = string
}