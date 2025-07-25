ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


# --- Base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    gnupg2 \
    lsb-release \
    locales \
    python3 \
    python3-pip \
    python3-full \
    python3-dev \
    sudo \
    x11-apps \
    zenity \
    openssh-client \
    wget \
    unzip \
    pkg-config \
    zip \
    libxinerama-dev \
    libxcursor-dev \
    xorg-dev \
    libglu1-mesa-dev \
    vim \
    nano \
    htop \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# --- User Setup
ARG USER_UID
ARG USER_GID
ARG USERNAME

RUN set -eux; \
# Remove user with same UID if exists
EXISTING_USER_BY_UID=$(getent passwd "${USER_UID}" | cut -d: -f1 || true); \
if [ -n "$EXISTING_USER_BY_UID" ]; then \
userdel -r "$EXISTING_USER_BY_UID" || true; \
fi; \
\
# Remove user with same name if exists
if id "${USERNAME}" >/dev/null 2>&1; then \
userdel -r "${USERNAME}" || true; \
fi; \
\
# Remove group with same GID if exists
if getent group "${USER_GID}" >/dev/null; then \
groupdel "$(getent group "${USER_GID}" | cut -d: -f1)"; \
fi; \
\
# Remove group with same name if exists
if getent group "${USERNAME}" >/dev/null; then \
groupdel "${USERNAME}"; \
fi; \
\
# Now create group and user cleanly
groupadd --gid "${USER_GID}" "${USERNAME}"; \
useradd -m -u "${USER_UID}" -g "${USER_GID}" "${USERNAME}"; \
usermod -aG sudo "${USERNAME}"; \
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set environment
ENV USER=${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR /home/${USERNAME}

USER ${USERNAME}

RUN git clone https://github.com/microsoft/vcpkg.git ${HOME}/vcpkg && \
    cd ${HOME}/vcpkg && ./bootstrap-vcpkg.sh -disableMetrics

RUN echo 'export VCPKG_ROOT=$HOME/vcpkg' >> $HOME/.bashrc && \
    echo 'export PATH=$VCPKG_ROOT:$PATH' >> $HOME/.bashrc

RUN wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu128.zip -d ${HOME} && \
    rm libtorch-cxx11-abi-shared-with-deps-2.7.1+cu128.zip

RUN wget https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-x86_64.sh && \
    chmod +x cmake-4.0.3-linux-x86_64.sh && \
    sudo ./cmake-4.0.3-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-4.0.3-linux-x86_64.sh

# Modified from https://github.com/jmhardison/dockercolorlogo
RUN echo '#!/bin/sh' >> ${HOME}/.docker_whale && \
    echo 'colorred="\033[31m"' >> ${HOME}/.docker_whale && \
    echo 'colorpowder_blue="\033[1;36m"' >> ${HOME}/.docker_whale && \
    echo 'colorblue="\033[34m"' >> ${HOME}/.docker_whale && \
    echo 'colornormal="\033[0m"' >> ${HOME}/.docker_whale && \
    echo 'colorwhite="\033[97m"' >> ${HOME}/.docker_whale && \
    echo 'colorlightgrey="\033[90m"' >> ${HOME}/.docker_whale && \
    echo '' >> ${HOME}/.docker_whale && \
    echo 'printf "                   ${colorred} ##       ${colorlightgrey} .         \n"' >> ${HOME}/.docker_whale && \
    echo 'printf "             ${colorred} ## ## ##      ${colorlightgrey} ==         \n"' >> ${HOME}/.docker_whale && \
    echo 'printf "           ${colorred}## ## ## ##      ${colorlightgrey}===         \n"' >> ${HOME}/.docker_whale && \
    echo 'printf "       /\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\\\___/ ===       \n"' >> ${HOME}/.docker_whale && \
    echo 'printf "  ${colorblue}~~~ ${colorlightgrey}{${colorblue}~~ ~~~~ ~~~ ~~~~ ~~ ~ ${colorlightgrey}/  ===- ${colorblue}~~~${colorlightgrey}\n"' >> ${HOME}/.docker_whale && \
    echo 'printf "       \\\______${colorwhite} o ${colorlightgrey}         __/           \n"' >> ${HOME}/.docker_whale && \
    echo 'printf "         \\\    \\\        __/            \n"' >> ${HOME}/.docker_whale && \
    echo 'printf "          \\\____\\\______/               \n"' >> ${HOME}/.docker_whale && \
    echo 'printf "\n${colorwhite}Welcome to gaussian-splatting-cuda dev container!${colornormal}\n"' >> ${HOME}/.docker_whale && \
    echo "bash \$HOME/.docker_whale" >> ${HOME}/.bashrc


ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]