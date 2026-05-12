# 🚀 Run Sentra using Docker

## ✅ I. Set Up NVIDIA Container Toolkit (Ubuntu)

You might need to install `Nvidia`'s container toolkit to enable GPU support in your Docker containers. Follow the instructions below to set it up on your Ubuntu system:

```bash
# Detect your Ubuntu distribution
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Add the NVIDIA GPG key
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the NVIDIA container repository
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure the Docker runtime (inside docker folder)
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

## ⚙️ II. Build and Run

Follow the commands below to build the Docker image and run the container:

```bash
# Build the Docker image
docker compose build

# Run the container
docker compose up -d

# Access the container
docker exec -it sentra_ros2 bash
```

Finally, inside the container, you can simply run `mprocs` to start the Sentra application.