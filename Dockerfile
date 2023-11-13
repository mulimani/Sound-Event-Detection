FROM python:3.10-slim
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
COPY . /root/SED_model
COPY requirements.txt .
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n sed-env && \
    conda activate sed-env && \
    conda install python=3.10 pip && \
    pip install -r requirements.txt
RUN echo "conda activate sed-env" >> /root/.bashrc
WORKDIR /root/SED_model
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["python main.py"]
#ENTRYPOINT ["./entrypoint.sh"]