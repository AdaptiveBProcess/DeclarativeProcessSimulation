FROM openjdk:8-jdk

# Install Xvfb and required libraries
RUN apt-get update && apt-get install -y \
    xvfb \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libgtk2.0-0 \
    libxext6 \
    libxrandr2 \
    libasound2 \
    libxinerama1 \
    libfreetype6 \
    && apt-get clean

# Set display env var
ENV DISPLAY=:99

# Set working directory
WORKDIR /app

# Copy your code into the container
COPY . /app

# Default CMD: run Xvfb + allow override
CMD Xvfb :99 -screen 0 1024x768x16 & java "$@"
