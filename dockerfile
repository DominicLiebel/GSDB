FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies for OpenSlide
RUN apt-get update && apt-get install -y \
    openslide-tools \
    python3-openslide \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Install the package
RUN pip install -e .

# Set environment variable for paths
ENV GASTRIC_BASE_DIR="/data"

# Default command to list available commands
CMD ["python", "-m", "slide_classification.list_commands"]