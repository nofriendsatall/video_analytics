#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Настройки
# ============================================================

# Название окружения можно передать первым аргументом:
# ./install.sh my-env
ENV_NAME="${1:-dino-track}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"

# Ставить ли supervision.
# Если нужен только детектор, можно:
# INSTALL_SUPERVISION=0 ./install.sh
INSTALL_SUPERVISION="${INSTALL_SUPERVISION:-1}"

# Ставить ли системные библиотеки для GUI через apt.
# Нужно, если хочешь использовать cv2.imshow в Linux/WSL.
# Для работы с флагом --no-display это не обязательно.
# Пример:
# INSTALL_APT_GUI_LIBS=1 ./install.sh
INSTALL_APT_GUI_LIBS="${INSTALL_APT_GUI_LIBS:-0}"

echo "=================================================="
echo "Conda env name:       ${ENV_NAME}"
echo "Python version:       ${PYTHON_VERSION}"
echo "CUDA version:         ${CUDA_VERSION}"
echo "Install supervision:  ${INSTALL_SUPERVISION}"
echo "Install apt GUI libs: ${INSTALL_APT_GUI_LIBS}"
echo "=================================================="

# ============================================================
# Проверка наличия conda
# ============================================================

if ! command -v conda >/dev/null 2>&1; then
    echo ""
    echo "Ошибка: conda не найден в PATH."
    echo "Установи Miniconda/Anaconda или активируй conda перед запуском скрипта."
    echo ""
    exit 1
fi

# ============================================================
# Создание или выбор окружения
# ============================================================

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo ""
    echo "Окружение ${ENV_NAME} уже существует."
    echo "Пакеты будут доустановлены/обновлены."
else
    echo ""
    echo "Создаю conda-окружение ${ENV_NAME}..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# ============================================================
# Обновление базовых инструментов pip
# ============================================================

echo ""
echo "=================================================="
echo "Установка/обновление pip, setuptools, wheel..."
echo "=================================================="

conda install -y -n "${ENV_NAME}" -c conda-forge \
    pip \
    setuptools \
    wheel

# ============================================================
# Опциональные системные библиотеки для GUI
# ============================================================

if [[ "${INSTALL_APT_GUI_LIBS}" == "1" ]]; then
    echo ""
    echo "=================================================="
    echo "Попытка установить системные библиотеки для GUI..."
    echo "=================================================="

    if command -v apt-get >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update || true
            sudo apt-get install -y \
                libgl1 \
                libglib2.0-0 \
                libgtk-3-0
        else
            apt-get update || true
            apt-get install -y \
                libgl1 \
                libglib2.0-0 \
                libgtk-3-0
        fi
    else
        echo "apt-get не найден. Пропускаю установку системных GUI-библиотек."
    fi
fi

# ============================================================
# Базовые пакеты для видео и изображений
# ============================================================

echo ""
echo "=================================================="
echo "Установка numpy, pillow, opencv, ffmpeg через conda..."
echo "=================================================="

conda install -y -n "${ENV_NAME}" -c conda-forge \
    numpy \
    pillow \
    opencv \
    ffmpeg

# ============================================================
# Установка PyTorch
# ============================================================

echo ""
echo "=================================================="
echo "Установка PyTorch..."
echo "=================================================="

if [[ "${CUDA_VERSION}" == "cpu" ]]; then
    echo "Устанавливаю CPU-версию PyTorch."

    conda install -y -n "${ENV_NAME}" -c pytorch \
        pytorch \
        torchvision \
        torchaudio \
        cpuonly
else
    echo "Пытаюсь установить PyTorch через conda с CUDA ${CUDA_VERSION}..."

    if conda install -y -n "${ENV_NAME}" -c pytorch -c nvidia \
        pytorch \
        torchvision \
        torchaudio \
        "pytorch-cuda=${CUDA_VERSION}"
    then
        echo "PyTorch успешно установлен через conda."
    else
        echo ""
        echo "=================================================="
        echo "Не удалось установить PyTorch через conda."
        echo "Пробую установить через pip с CUDA 12.8..."
        echo "=================================================="

        conda run -n "${ENV_NAME}" python -m pip install -U pip

        # Для новых GPU, например RTX 5060 Ti, иногда нужна свежая сборка.
        conda run -n "${ENV_NAME}" python -m pip install --pre \
            torch \
            torchvision \
            --index-url https://download.pytorch.org/whl/cu128
    fi
fi

# ============================================================
# Установка зависимостей для Grounding DINO
# ============================================================

echo ""
echo "=================================================="
echo "Установка зависимостей для Grounding DINO..."
echo "=================================================="

conda run -n "${ENV_NAME}" python -m pip install -U \
    pip \
    setuptools \
    wheel

conda run -n "${ENV_NAME}" python -m pip install -U \
    "transformers>=4.49" \
    accelerate \
    timm \
    "huggingface_hub[cli]"

# ============================================================
# Установка supervision, если нужен трекер
# ============================================================

if [[ "${INSTALL_SUPERVISION}" == "1" ]]; then
    echo ""
    echo "=================================================="
    echo "Установка supervision..."
    echo "=================================================="

    conda run -n "${ENV_NAME}" python -m pip install -U supervision
else
    echo ""
    echo "supervision пропущен, так как INSTALL_SUPERVISION=${INSTALL_SUPERVISION}"
fi

# ============================================================
# Проверка установки
# ============================================================

echo ""
echo "=================================================="
echo "Проверка установленных пакетов..."
echo "=================================================="

conda run -n "${ENV_NAME}" python - <<'PY'
import importlib.util

import numpy as np
import torch
import transformers
import cv2
import PIL

print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Внимание: torch.cuda.is_available() == False")
    print("Если у тебя есть NVIDIA GPU, проверь драйвер и версию PyTorch.")

print("transformers version:", transformers.__version__)
print("OpenCV version:", cv2.__version__)
print("Pillow version:", PIL.__version__)

if importlib.util.find_spec("supervision") is not None:
    import supervision as sv
    print("supervision version:", sv.__version__)
else:
    print("supervision: not installed")
PY

echo ""
echo "=================================================="
echo "Готово!"
echo ""
echo "Активируй окружение командой:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "Пример запуска детекции:"
echo "    python temp.py \\"
echo "      --source input.mp4 \\"
echo "      --output output.mp4 \\"
echo "      --model-id IDEA-Research/grounding-dino-base \\"
echo "      --device cuda \\"
echo "      --fp16 \\"
echo "      --max-size 1280 \\"
echo "      --no-display"
echo "=================================================="
