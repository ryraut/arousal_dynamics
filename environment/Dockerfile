# hash:sha256:91e10f9a790809103e5bdda9b64bb7c1869923ed7b5b559490f5862cdc45de32
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==3.3.2 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    ipympl==0.9.3 \
    keras==2.10.0 \
    mat73==0.62 \
    matplotlib==3.8.3 \
    numpy==1.26.4 \
    scikit-learn==1.4.1.post1 \
    scipy==1.12.0 \
    tensorflow==2.10.0
