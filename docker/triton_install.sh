#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced. Run 'source $(basename "${BASH_SOURCE[0]}")'" >&2
    exit 1
fi

# Save original positional parameters and OPTIND
original_params=("$@")
original_OPTIND=$OPTIND
OPTIND=1
restore_env() {
    set -- "${original_params[@]}"
    OPTIND=$original_OPTIND
}
trap restore_env EXIT

show_help() {
    echo "Usage: source $(basename "${BASH_SOURCE[0]}") [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -g, --git-commit=HASH     Check out specific git commit"
    echo "  -l, --llvm-build-dir=DIR  Use custom LLVM build directory"
    echo "  -c, --clang-path=DIR      Path to directory containing clang/clang++"
    echo "                            (default: \${ROCM_PATH}/llvm/bin)"
    echo "                            Only required when using custom LLVM build"
    return 0
}

patch_triton_source() {
    local file="python/triton/backends/amd/compiler.py"
    echo "Attempting to patch ${file}..."
    if [ -f "${file}" ]; then
        if grep -q "^[[:space:]]*assert len(names) == 1" "${file}"; then
            sed -i 's/^[[:space:]]*assert len(names) == 1/#        assert len(names) == 1/' "${file}"
            echo "Successfully patched ${file}"
            return 0
        else
            echo "Assert statement not found in ${file}, continuing without patch"
            return 0
        fi
    else
        echo "Warning: ${file} not found, continuing without patch"
        return 0
    fi
}

OPTIND=1
# Process command line parameters
while getopts "g:l:c:h-:" opt; do
    case $opt in
        h)
            show_help
            return 0
            ;;
        g)
            COMMIT="$OPTARG"
            ;;
        l)
            LLVM_BUILD_DIR="$OPTARG"
            ;;
        c)
            CLANG_PATH="$OPTARG"
            ;;
        -)
            case "${OPTARG}" in
                help)
                    show_help
                    return 0
                    ;;
                git-commit=*)
                    COMMIT="${OPTARG#*=}"
                    ;;
                llvm-build-dir=*)
                    LLVM_BUILD_DIR="${OPTARG#*=}"
                    ;;
                clang-path=*)
                    CLANG_PATH="${OPTARG#*=}"
                    ;;
                *)
                    echo "Invalid option: --${OPTARG}" >&2
                    return 1
                    ;;
            esac
            ;;
    esac
done
shift $((OPTIND - 1))

git clone https://github.com/triton-lang/triton.git
cd triton

# If commit parameter is provided, check out that commit.
if [ -n "${COMMIT}" ]; then
    git checkout "${COMMIT}"
fi

python -m venv .venv --prompt triton
source .venv/bin/activate
pip install ninja cmake wheel pybind11 # build-time dependencies
pip install matplotlib pandas # run-time dependencies
# install PyTorch, but remove the PyTorch Triton extension
# to avoid conflicts with the Triton build
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip uninstall --yes pytorch-triton-rocm

# Conditional LLVM
if [ -n "${LLVM_BUILD_DIR}" ]; then
    # Set default clang path if not specified
    if [ -z "${CLANG_PATH}" ] && [ -n "${ROCM_PATH}" ]; then
        CLANG_PATH="${ROCM_PATH}/llvm/bin"
    fi

    if [ -n "${CLANG_PATH}" ]; then
        CLANG="${CLANG_PATH}/clang"
        CLANGXX="${CLANG_PATH}/clang++"
        if [ ! -x "${CLANG}" ] || [ ! -x "${CLANGXX}" ]; then
            echo "Warning: Clang compilers not found in ${CLANG_PATH}" >&2
            echo "Build might fail. Check if the path is correct." >&2
        else
            export CC="${CLANG}"
            export CXX="${CLANGXX}"
        fi
    fi

    echo "Using custom LLVM build directory: ${LLVM_BUILD_DIR}"
    export PATH="${LLVM_BUILD_DIR}/bin:${PATH}"
    export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}:${LD_LIBRARY_PATH}"
    export LLVM_INCLUDE_DIRS="${LLVM_BUILD_DIR}/include"
    export LLVM_LIBRARY_DIR="${LLVM_BUILD_DIR}/lib"
    export LLVM_SYSPATH="${LLVM_BUILD_DIR}"
fi


# Build and install Triton
pip install -e python

PIP_STATUS=$?
if [ $PIP_STATUS -eq 0 ]; then
    # Remove assertion in compiler.py that doesn't hold when cloning kernels
    patch_triton_source
    if [ -z "${ROCM_PATH}" ] || [ ! -x "${ROCM_PATH}/llvm/bin/ld.lld" ]; then
        echo "Triton installed successfully!"
        echo "Note: TRITON_HIP_LLD_PATH needs to be set manually to the path of ld.lld"
        echo "This is typically found in one of:"
        echo "  - \${ROCM_PATH}/llvm/bin/ld.lld"
        echo "  - /opt/rocm/llvm/bin/ld.lld"
        echo "  - /opt/rocm-\{version\}/llvm/bin/ld.lld"
        return 0
    else
        export TRITON_HIP_LLD_PATH="${ROCM_PATH}/llvm/bin/ld.lld"
        echo "Set TRITON_HIP_LLD_PATH to ${TRITON_HIP_LLD_PATH}"
        echo "Triton installed successfully!"
    fi
else
    echo "Error: Triton installation failed" >&2
    return $PIP_STATUS
fi