FROM tensorflow/tensorflow:2.9.1-gpu

LABEL maintainer="abhilash"
LABEL version="2.0"
LABEL description="Docker environment for DQN-ASV-path-follow"

#RUN apt-get update
RUN apt-get -y install software-properties-common libsm6 libxext6 libxrender1 libfontconfig1 libgl1 sudo xauth
#RUN apt-get update

RUN pip install --upgrade pip
COPY ./requirements_tf_291.txt /requirements.txt
RUN pip install -r requirements.txt

ENV USERNAME docker
RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME 

WORKDIR /home/docker
USER docker

CMD ["bash"]
