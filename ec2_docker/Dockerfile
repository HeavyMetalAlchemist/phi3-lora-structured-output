FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements_ec2.txt .

RUN pip install --no-cache-dir -r requirements_ec2.txt

CMD ["/bin/bash"]