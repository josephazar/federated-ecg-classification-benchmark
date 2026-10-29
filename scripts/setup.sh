#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"
if [ ! -x .venv/bin/python ]; then
    "${PYTHON_BOOTSTRAP:-python3.12}" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
mkdir -p vendor
if [ ! -d vendor/liboqs ]; then
    git clone https://github.com/open-quantum-safe/liboqs.git vendor/liboqs
    git -C vendor/liboqs checkout 5a1a854b0dc9f2141bdc771c555ee60c37950183
fi
PATH="$PROJECT_ROOT/.venv/bin:$PATH" .venv/bin/cmake -S vendor/liboqs -B vendor/liboqs/build -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOQS_BUILD_ONLY_LIB=ON -DOQS_USE_OPENSSL=OFF \
    '-DOQS_MINIMAL_BUILD=KEM_ml_kem_512;KEM_ml_kem_768;KEM_ml_kem_1024;SIG_ml_dsa_44;SIG_ml_dsa_65;SIG_ml_dsa_87' \
    -DCMAKE_INSTALL_PREFIX="$PROJECT_ROOT/vendor/oqs-install"
.venv/bin/cmake --build vendor/liboqs/build --parallel 8
.venv/bin/cmake --install vendor/liboqs/build
printf '%s\n' 'Ready. Use .venv/bin/python for all experiment commands.'
