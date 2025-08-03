FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /app
COPY . /app
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install deepbrain --no-deps
RUN rm -rf /root/.cache/pip/*
RUN sed -i '1s/.*/import tensorflow.compat.v1 as tf/' /usr/local/lib/python3.11/dist-packages/deepbrain/extractor.py