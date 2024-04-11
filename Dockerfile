# Pytorch image
FROM python:3.11
RUN pip install --upgrade pip
RUN pip install torch torchvision

# Install git and miniconda
RUN apt-get update && apt-get install -y git wget

# Set the working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://www.github.com/abelsr/Neural_Vlasov_Maxwell

# Check pytorch installation
RUN python -c "import torch; print(torch.__version__)"

# Install the required packages
RUN pip install -r Neural_Vlasov_Maxwell/requirements.txt

# Set the image entrypoint as the bash shell
CMD ["/bin/bash"]