FROM python:3.8.3
WORKDIR /app
COPY . /app
RUN pip config set global.index-url https://mirrors.zju.edu.cn/pypi/web/simple
RUN pip install -r requirements.txt
RUN pip install deepbrain --no-deps
RUN rm -rf /root/.cache/pip/*
RUN sed -i '1s/.*/import tensorflow.compat.v1 as tf/' /usr/local/lib/python3.8/site-packages/deepbrain/extractor.py