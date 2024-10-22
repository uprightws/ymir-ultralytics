# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
FROM ultralytics/ultralytics
# support YMIR=1.0.0, 1.1.0 or 1.2.0
ARG YMIR="2.1.0"


ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV LANG=C.UTF-8
ENV YMIR_VERSION=$YMIR
ENV YOLOV5_CONFIG_DIR='/app/data'

# Install linux package
# RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
#     libgl1-mesa-glx libsm6 libxext6 libxrender-dev curl wget zip vim \
#     build-essential ninja-build \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
# RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.1.0"


# Copy file from host to docker and install requirements
COPY . /app
RUN pip install .
RUN mkdir /img-man && mv /app/ymir/img-man/*.yaml /img-man/

# install and requirements
# RUN pip install -r /app/requirements.txt 
# Download pretrained weight and font file
# RUN cd /app && pip install -e .
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt /app
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt /app
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt /app
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10l.pt /app
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# install ymir-exc sdk


# Make PYTHONPATH find local package
ENV PYTHONPATH=.

WORKDIR /app
RUN echo "python3 /app/ymir/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh