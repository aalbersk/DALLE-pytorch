
ARG IMG_TAG=1.9.0-cuda10.2-cudnn7-devel
ARG IMG_REPO=pytorch

FROM pytorch/$IMG_REPO:$IMG_TAG

COPY ./16L_64HD_8H_512I_128T_cc12m_cc3m_3E.pt 16L_64HD_8H_512I_128T_cc12m_cc3m_3E.pt
COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb

RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get -y update && apt-get -y install git gcc llvm-9-dev cmake libaio-dev vim wget

#RUN git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
#RUN cd /tmp/DeepSpeed && DS_BUILD_OPS=1 ./install.sh -r
#RUN pip install git+https://github.com/lucidrains/DALLE-pytorch.git

WORKDIR dalle
