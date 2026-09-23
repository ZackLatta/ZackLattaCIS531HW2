FROM python:3.10-slim-bookworm

WORKDIR /workspace


RUN python -m pip install --no-cache-dir --upgrade pip setuptools

RUN python -m pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cu121

COPY probe_compute.py .

CMD ["python", "probe_compute.py"]