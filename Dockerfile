FROM python:latest

WORKDIR /pacbot

COPY . .

# Install dependencies for stable-baselines 3
RUN apt-get update && apt-get -y install cmake libopenmpi-dev python3-dev zlib1g-dev

# Install packages
RUN pip install -r requirements.txt

CMD ["python3", "main.py"]

