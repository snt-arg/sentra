FROM nvcr.io/nvidia/cuda-dl-base:25.04-cuda12.9-devel-ubuntu24.04

# ----------
# Arguments
# ----------
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# ----------------------
# Environment Variables
# ----------------------
ENV CUDA_HOME=/usr/local/cuda \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=jazzy \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ----------------------
# Fix MPI Library issue
# ----------------------
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu \
    && ln -s /opt/hpcx/ompi /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi \
    && dpkg-reconfigure libc-bin

# --------------
# Create a User
# --------------
RUN if id -u $USER_UID ; then userdel "$(id -un $USER_UID)" ; fi

# --------------
# System Setup
# --------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python-is-python3 \
    git \
    openssh-client \
    wget \
    vim \
    curl \
    cmake \
    build-essential \
    locales \
    software-properties-common \
    lsb-release \
    gnupg2 && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN add-apt-repository universe

# -------------------------------------
# Install ROS 2 Jazzy and Dependencies
# -------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    apt install -y /tmp/ros2-apt-source.deb
RUN apt update && apt upgrade -y && \
    apt install -y ros-dev-tools \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-rqt-tf-tree \
    ros-${ROS_DISTRO}-rviz-visual-tools \
    ros-${ROS_DISTRO}-depth-image-proc \
    ros-${ROS_DISTRO}-backward-ros \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-diagnostic-updater \
    ros-${ROS_DISTRO}-pcl-ros
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc
RUN rosdep init && rosdep update

# -----------------------
# Clean up the apt Cache
# -----------------------
RUN rm -rf /var/lib/apt/lists/* /tmp/*

# --------------
# Create a User
# --------------
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# -----------------------
# Setup Python Libraries
# -----------------------
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch \
    torchvision \
    networkx==3.1
RUN apt remove --purge python3-typing-extensions -y
RUN pip3 install typing-extensions==4.11.0

# ---------
# SSH Keys
# ---------
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# ----------------
# Workspace Setup
# ----------------
WORKDIR /home/$USERNAME/workspace/src
RUN --mount=type=ssh git clone git@github.com:snt-arg/sentra.git
WORKDIR /home/$USERNAME/workspace/src/sentra
RUN pip3 install --no-cache-dir -r sentra_ros/requirements.txt

# ----------------
# Build Workspace
# ----------------
WORKDIR /home/$USERNAME/workspace/
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# ---------
# Clean Up
# ---------
RUN ldconfig
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean && apt-get autoremove -y
RUN rm -rf /root/.ssh/

# -----------------
# Build Entrypoint
# -----------------
RUN echo "#!/bin/bash" >> /entrypoint.sh \
    && echo "echo \"source /opt/ros/$ROS_DISTRO/setup.bash\" >> ~/.bashrc" >> /entrypoint.sh \
    && echo "echo \"source /home/$USERNAME/workspace/install/setup.bash\" >> ~/.bashrc" >> /entrypoint.sh \
    && echo 'exec "$@"' >> /entrypoint.sh \
    && chmod a+x /entrypoint.sh

# -----------------
# Change Ownership
# -----------------
USER $USERNAME
RUN sudo chown -R $USERNAME:$USERNAME /home/$USERNAME/workspace
WORKDIR /home/$USERNAME/workspace/

# --------
# Mprocs
# --------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y \
    && . "$HOME/.cargo/env" \
    && cargo install mprocs

# ------------------
# Environment Setup
# ------------------
RUN echo "alias mprocs='mprocs -c /home/$USERNAME/workspace/src/sentra/config/mprocs.yml'" >> ~/.bashrc

# ------------------
# Entrypoint and CMD
# ------------------
ENTRYPOINT ["/entrypoint.sh"]
USER $USERNAME
CMD ["/bin/bash"]
SHELL ["/bin/bash"]