FROM python:3.7

WORKDIR /

COPY . /

RUN pip install sacrebleu

CMD ["/bin/bash"]
