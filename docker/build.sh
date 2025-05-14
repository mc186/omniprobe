#!/bin/bash

# Container name
name="logduration"

# Script directories
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"
cur_dir=$(pwd)

# Parse arguments
build_docker=false
build_apptainer=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --apptainer)
      build_apptainer=true
      shift
      ;;
    --docker)
      build_docker=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--docker] [--apptainer] -- At least one option is required."
      exit 1
      ;;
  esac
done

if [ "$build_docker" = false ] && [ "$build_apptainer" = false ]; then
    echo "Error: At least one of the options --docker or --apptainer is required."
    echo "Usage: $0 [--docker] [--apptainer]"
    exit 1
fi

pushd "$script_dir"
  
if [ "$build_docker" = true ]; then
    echo "Building Docker container..."

    # Auto-configure SSH agent
    if [ ! -S ~/.ssh/ssh_auth_sock ]; then
        eval "$(ssh-agent)" > /dev/null
        ln -sf "$SSH_AUTH_SOCK" ~/.ssh/ssh_auth_sock
    fi
    export SSH_AUTH_SOCK=~/.ssh/ssh_auth_sock

    # Add default keys if they exist
    [ -f ~/.ssh/id_rsa ] && ssh-add ~/.ssh/id_rsa
    [ -f ~/.ssh/id_ed25519 ] && ssh-add ~/.ssh/id_ed25519
    [ -f ~/.ssh/id_github ] && ssh-add ~/.ssh/id_github

    # Enable BuildKit and build the Docker image
    export DOCKER_BUILDKIT=1
    docker build \
        --ssh default \
        -t "$name" \
        -f "$script_dir/logduration.Dockerfile" \
        .

    echo "Docker build complete!"
fi

if [ "$build_apptainer" = true ]; then
    echo "Building Apptainer container..."

    # Check if apptainer is installed
    if ! command -v apptainer &> /dev/null; then
        echo "Error: Apptainer is not installed or not in PATH"
        echo "Please install Apptainer first: https://apptainer.org/docs/admin/main/installation.html"
        exit 1
    fi

    # Build the Apptainer container
    apptainer build "${name}.sif" "$script_dir/logduration.def"

    echo "Apptainer build complete!"
fi
  
popd