FROM ubuntu:20.04

# Disable apt auto-cleanup to enable caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Install Python and setup virtual environment
WORKDIR /
RUN --mount=type=cache,target=/var/cache/apt,id=apt \
    --mount=type=cache,target=/var/lib/apt/lists,id=apt-list \
    DEBIAN_FRONTEND=noninteractive apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
        git \
        vim \
        wget \
        unzip \
        python3.8-dev \
        python3-pip

ENV VIRTUAL_ENV=/opt/venv \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=secret,id=pipconfig,dst=/etc/pip.conf \
    --mount=type=cache,target=/root/.cache/pip,id=pip \
    python3.8 -m pip install --upgrade pip virtualenv \
    && python3.8 -m virtualenv "$VIRTUAL_ENV"

# From now on, python will point to our virtualenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Mujoco, see https://github.com/deepmind/dm_control
RUN --mount=type=cache,target=/var/cache/apt,id=apt \
    --mount=type=cache,target=/var/lib/apt/lists,id=apt-list \
    DEBIAN_FRONTEND=noninteractive apt install -y \
      # Software-rendering only
      libgl1-mesa-glx \
      libosmesa6-dev \
      libglfw3 \
      patchelf \
      xvfb \
      freeglut3-dev

WORKDIR /opt
RUN wget -q --show-progress "https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz" -O mujoco.tar.gz \
    && tar -xzf mujoco.tar.gz \
    && rm mujoco.tar.gz
ENV MJLIB_PATH="/opt/mujoco210/bin/libmujoco210.so" \
	MUJOCO_PY_MUJOCO_PATH="/opt/mujoco210" \
    MUJOCO_GL="osmesa" \
	LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/mujoco210/bin"

# Install pytorch with CUDA support
WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip,id=pip \
    python -m pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install mujoco-py
RUN --mount=type=cache,target=/root/.cache/pip,id=pip \
    python -m pip install mujoco-py

# Install packages in editable mode
ENV REPOS_DIR /root/repos
WORKDIR $REPOS_DIR
COPY . $REPOS_DIR/ube-mbrl/
RUN --mount=type=cache,target=/root/.cache/pip,id=pip \
    python -m pip install -e ube-mbrl/ 

# Verify installation
COPY ./docker/test_install.sh /tmp/test_install.sh
RUN chmod +x /tmp/test_install.sh && /tmp/test_install.sh
