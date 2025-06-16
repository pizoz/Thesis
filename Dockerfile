FROM python:3.11-bookworm

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install -r requirements.txt

