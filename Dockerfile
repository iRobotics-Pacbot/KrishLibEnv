FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /pacbot

COPY . .

# Install packages
RUN apt update
RUN apt install -y python3-pip
RUN pip install -r requirements.txt

CMD ["python3", "train.py"]

