FROM python:3.7-alpine

RUN apt-get update && apt-get install graphviz git wget gfortran libopenblas-dev liblapack-dev -y


# for non-root user

ARG APP_TOKEN

# env vars
#ENV USER=root
#ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1




COPY requirements.txt /tmp/requirements.txt

RUN  pip install  -r /tmp/requirements.txt
# directory for following operations
WORKDIR /forecast_bcl

COPY . .

RUN pip install -e .
