# Use the official R-devel image (Debian-based)
FROM rocker/r-devel

# Install system dependencies (adjust this if your package needs libssl, libxml2, etc.)
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    zlib1g-dev

# Create a directory for your package
WORKDIR /home/cvLM

# Copy your local package code into the container
COPY . .

# Install dependencies and the package itself
RUN R -e "install.packages('remotes', repos='https://cloud.r-project.org')"
RUN R -e "remotes::install_deps(dependencies = TRUE)"

# Set the default command to run the CRAN check
CMD ["R", "CMD", "check", "--as-cran", "."]