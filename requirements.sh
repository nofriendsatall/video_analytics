#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Настройки установки
# ============================================================

# Имя окружения
ENV_NAME="${ENV_NAME:-dino-boxmot}"

# Версия Python
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Вариант PyTorch:
#   cu128 - CUDA 12.8
#   cu126 - CUDA 12.6
#   cu124 - CUDA 12.4
#   cpu   - CPU-версия
#
# По умолчанию ставится версия под CUDA 12.8.
CUDA_VARIANT="${CUDA_VARIANT:-cu128}"

# Если хочешь принудительно поставить CPU-версию:
#   CPU=1 bash install_cuda128.sh
CPU="${CPU:-0}"

# Если хочешь использовать ночную сборку PyTorch для CUDA 12.8:
#   PYTORCH_NIGHTLY=1 bash install_cuda128.sh
PYTORCH_NIGHTLY="${PYTORCH_NIGHTLY:-0}"

# Если WITH_GUI=1, ставится обычный opencv-python.
# Если WITH_GUI=0, ставится opencv-python-headless.
#
# Для сервера/без дисплея лучше:
#   WITH_GUI=0
#
# Если хочешь проверять окно через cv2.imshow:
#   WITH_GUI=1
WITH_GUI="${WITH_GUI:-0}"

# Версия boxmot.
# Если хочешь самую свежую:
#   BOXMOT_PACKAGE=boxmot bash install_cuda128.sh
#
# Если хочешь из репозитория:
#   BOXMOT_PACKAGE="git+https://github.com/mikel-brostrom/boxmot.git" bash install_cuda128.sh
BOXMOT_PACKAGE="${BOXMOT_PACKAGE:-boxmot==25.0.0}"

# Ставить ли ultralytics.
# Некоторым версиям boxmot он нужен как зависимость.
# Если не нужен, можно отключить:
#   INSTALL_ULTRALYTICS=0 bash install_cuda128.sh
INSTALL_ULTRALYTICS="${INSTALL_ULTRALYTICS:-1}"

# Скачать ли модель Grounding DINO после установки.
# По умолчанию не скачиваем, чтобы не тянуть веса во время установки.
# Если хочешь скачать сразу:
#   DOWNLOAD_MODEL=1 bash install_cuda128.sh
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_ID="${MODEL_ID:-IDEA-Research/grounding-dino-tiny}"

# ============================================================
# Вспомогательные функции
# ============================================================

log() {
    echo -e "\n\033[1;32m[$(date +'%H:%M:%S')] $*\033[0m"
}

warn() {
    echo -e "\033[1;33m[$(date +'%H:%M:%S')] $*\033[0m"
}

die() {
    echo -e "\n\033[1;31m[ERROR] $*\033[0m" >&2
    exit 1
}

# ============================================================
# Проверка окружения
# ============================================================

if [[ "$CPU" == "1" ]]; then
    CUDA_VARIANT="cpu"
fi

log "Параметры установки:"
echo "ENV_NAME:            ${ENV_NAME}"
echo "PYTHON_VERSION:      ${PYTHON_VERSION}"
echo "CUDA_VARIANT:        ${CUDA_VARIANT}"
echo "PYTORCH_NIGHTLY:     ${PYTORCH_NIGHTLY}"
echo "WITH_GUI:            ${WITH_GUI}"
echo "BOXMOT_PACKAGE:      ${BOXMOT_PACKAGE}"
echo "INSTALL_ULTRALYTICS: ${INSTALL_ULTRALYTICS}"
echo "DOWNLOAD_MODEL:      ${DOWNLOAD_MODEL}"
echo "MODEL_ID:            ${MODEL_ID}"

if ! command -v conda >/dev/null 2>&1; then
    die "conda не найден. Установи Miniconda или Anaconda."
fi

# Подключаем conda для текущего shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# ============================================================
# Создание окружения
# ============================================================

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "Создаю окружение ${ENV_NAME} с Python ${PYTHON_VERSION}"
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
else
    log "Окружение ${ENV_NAME} уже существует"
fi

conda activate "${ENV_NAME}"

log "Используется Python:"
python --version

# ============================================================
# Базовые инструменты
# ============================================================

log "Обновляю pip, setuptools и wheel"

python -m pip install --upgrade pip

# setuptools нужен для старых пакетов, которые используют pkg_resources.
# Ограничение <81 помогает сохранить совместимость с некоторыми старыми библиотеками.
python -m pip install --upgrade "setuptools<81" wheel

# ============================================================
# PyTorch для CUDA 12.8
# ============================================================

log "Устанавливаю PyTorch (${CUDA_VARIANT})"

if [[ "${CUDA_VARIANT}" == "cpu" ]]; then
    python -m pip install --upgrade torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

elif [[ "${PYTORCH_NIGHTLY}" == "1" ]]; then
    warn "Использую ночную сборку PyTorch для ${CUDA_VARIANT}"
    python -m pip install --pre --upgrade torch torchvision \
        --index-url "https://download.pytorch.org/whl/nightly/${CUDA_VARIANT}"

else
    python -m pip install --upgrade torch torchvision \
        --index-url "https://download.pytorch.org/whl/${CUDA_VARIANT}" || {
        warn "Не удалось установить стабильную версию из индекса ${CUDA_VARIANT}."
        warn "Если нужная версия ещё не опубликована в стабильном канале, попробуй ночную сборку:"
        warn "  PYTORCH_NIGHTLY=1 bash install_cuda128.sh"
        die "Установка PyTorch для ${CUDA_VARIANT} завершилась с ошибкой."
    }
fi

# ============================================================
# Проверка версии CUDA в PyTorch
# ============================================================

log "Проверяю, какую CUDA видит PyTorch"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    print("GPU недоступен или установлена CPU-версия.")
PY

# ============================================================
# NumPy и OpenCV
# ============================================================

log "Устанавливаю numpy"

# numpy<2 часто безопаснее для CV-библиотек и старых пакетов.
# Если позже зависимости будут требовать numpy>=2, можно переопределить:
#   NUMPY_SPEC="numpy" bash install_cuda128.sh
NUMPY_SPEC="${NUMPY_SPEC:-numpy>=1.26,<2}"

python -m pip install --upgrade "${NUMPY_SPEC}"

log "Устанавливаю OpenCV"

if [[ "${WITH_GUI}" == "1" ]]; then
    python -m pip install --upgrade opencv-python
else
    # headless-версия лучше подходит для серверов и контейнеров.
    # Для отображения окон через cv2.imshow лучше ставить:
    #   WITH_GUI=1
    python -m pip install --upgrade opencv-python-headless
fi

# ============================================================
# Зависимости для Grounding DINO / Hugging Face
# ============================================================

log "Устанавливаю зависимости для Hugging Face Transformers"

python -m pip install --upgrade \
    "transformers>=4.46" \
    accelerate \
    safetensors \
    huggingface_hub \
    tokenizers \
    Pillow \
    requests \
    tqdm \
    pyyaml \
    scipy \
    packaging

# ============================================================
# Зависимости трекинга
# ============================================================

log "Устанавливаю supervision"

python -m pip install --upgrade supervision

# ============================================================
# ВАЖНО: Устанавливаем совместимую версию gdown ДО установки
# любых пакетов, которые могут его использовать.
#
# Ошибка "download() got an unexpected keyword argument 'fuzzy'"
# возникает, когда установлена версия gdown, не поддерживающая
# аргумент 'fuzzy'.
#
# Совместимые версии:
#   - >= 4.6.0 и < 5.0.0 — поддерживают аргумент 'fuzzy'
#   - >= 5.0.0 — аргумент 'fuzzy' удалён
#
# Поэтому фиксируем версию 4.7.3 (последняя стабильная в ветке 4.7).
# ============================================================

log "Устанавливаю совместимую версию gdown (4.7.3)"

# Сначала удаляем любую существующую версию, чтобы избежать конфликтов
python -m pip uninstall -y gdown 2>/dev/null || true

# Устанавливаем фиксированную версию
python -m pip install --no-cache-dir "gdown==4.7.3"

log "Проверяю версию gdown"

python - <<'PY'
import gdown
import inspect

print("gdown version:", gdown.__version__)

# Проверяем сигнатуру функции download
sig = inspect.signature(gdown.download)
params = sig.parameters
print("gdown.download parameters:", list(params.keys()))

if "fuzzy" in params:
    print("OK: аргумент 'fuzzy' поддерживается")
else:
    print("ERROR: аргумент 'fuzzy' НЕ поддерживается")
    print("Установите версию: pip install gdown==4.7.3")
    exit(1)
PY

# ============================================================
# Устанавливаем boxmot
# ============================================================

log "Устанавливаю ${BOXMOT_PACKAGE}"

python -m pip install --upgrade "${BOXMOT_PACKAGE}" || {
    warn "Не удалось установить ${BOXMOT_PACKAGE}. Пробую поставить просто boxmot."
    python -m pip install --upgrade boxmot
}

# Проверка, что пакет реально установился
if ! python -m pip show boxmot >/dev/null 2>&1; then
    die "boxmot не был установлен."
fi

log "Устанавливаю дополнительные зависимости для трекеров"

python -m pip install --upgrade lapx || warn "lapx не установился. Если трекеры будут падать, попробуй: python -m pip install lapx"

# Иногда может быть полезен для совместимости со старыми трекерными сборками.
python -m pip install --upgrade cython || warn "cython не установился."

if [[ "${INSTALL_ULTRALYTICS}" == "1" ]]; then
    log "Устанавливаю ultralytics"
    python -m pip install --upgrade ultralytics || warn "ultralytics не установился."
fi

# ============================================================
# Проверка импортов
# ============================================================

log "Проверяю базовые импорты"

python - <<'PY'
import importlib

modules = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "PIL",
    "transformers",
    "supervision",
    "boxmot",
    "gdown",
    "lapx",
]

failed = []

for name in modules:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK   {name:<15} {version}")
    except Exception as e:
        print(f"FAIL {name:<15} {e}")
        failed.append(name)

if failed:
    print(f"\nНе удалось импортировать модули: {', '.join(failed)}")
    print("Это не всегда критично, но стоит проверить.")
PY

# ============================================================
# Проверка CUDA после установки всех пакетов
# ============================================================

log "Финальная проверка CUDA"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("GPU memory total:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("GPU недоступен или установлена CPU-версия.")
PY

# ============================================================
# Проверка структуры boxmot
# ============================================================

log "Проверяю структуру установленного boxmot"

python - <<'PY'
try:
    import os
    import boxmot

    print("boxmot version:", getattr(boxmot, "__version__", "unknown"))

    try:
        import boxmot.trackers.box
        box_path = os.path.join(
            os.path.dirname(boxmot.__file__),
            "trackers",
            "box"
        )

        print("box trackers path:", box_path)

        if os.path.isdir(box_path):
            trackers = sorted(os.listdir(box_path))
            print("Available trackers:", trackers)
        else:
            print("Папка с трекерами не найдена. Возможно, установлена старая версия.")
    except Exception as e:
        print("Проверка структуры трекеров завершилась ошибкой:", repr(e))

except Exception as e:
    print("boxmot check failed:", repr(e))
    raise
PY

# ============================================================
# Проверка зависимостей на конфликты
# ============================================================

log "Проверяю совместимость установленных пакетов"

python -m pip check || warn "pip check обнаружил несовместимости. Это не всегда критично, но стоит проверить."

# ============================================================
# Опциональная загрузка модели
# ============================================================

if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
    log "Скачиваю модель ${MODEL_ID}"

    MODEL_ID="${MODEL_ID}" python - <<'PY'
import os

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = os.environ["MODEL_ID"]

print("Загрузка:", model_id)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id,
    trust_remote_code=True
)

print("Модель успешно загружена.")
PY
else
    log "Модель не скачивается. При первом запуске она скачается автоматически."
fi

# ============================================================
# Инструкция по весам ReID
# ============================================================

log "Информация о весах ReID"

echo ""
echo "=========================================="
echo "ВЕСА REID НЕ СКАЧИВАЮТСЯ АВТОМАТИЧЕСКИ"
echo "=========================================="
echo ""
echo "Для работы трекеров с ReID (StrongSort, BotSort, OccluBoost, DeepOCSORT, HybridSORT)"
echo "нужно вручную скачать веса модели:"
echo ""
echo "  Название:  osnet_x1_0_msmt17.pt"
echo "  Ссылка:    https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"
echo "  Размер:    ~240 МБ"
echo ""
echo "Куда положить файл:"
echo ""
echo "  Вариант 1: рядом со скриптом (в текущей директории)"
echo "    ./osnet_x1_0_msmt17.pt"
echo ""
echo "  Вариант 2: в системную папку моделей"
echo "    $(python -c 'import site; print(site.getsitepackages()[0])')/models/osnet_x1_0_msmt17.pt"
echo ""
echo "Как скачать:"
echo ""
echo "  Через браузер:"
echo "    https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"
echo ""
echo "  Через gdown (рекомендуется):"
echo "    python -m gdown https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M -O osnet_x1_0_msmt17.pt"
echo ""
echo "  Через wget (если gdown не работает):"
echo "    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M' -O osnet_x1_0_msmt17.pt"
echo ""
echo "После скачивания запускай скрипт с явным путём к весам:"
echo ""
echo "  python temp.py --source video.mp4 --output out.mp4 \\"
echo "      --tracker strongsort \\"
echo "      --reid-weights ./osnet_x1_0_msmt17.pt"
echo ""
echo "=========================================="

# ============================================================
# Финальные инструкции
# ============================================================

log "Установка завершена."

echo ""
echo "Активация окружения:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Пример запуска:"
echo "  python temp.py --source video.mp4 --output out.mp4 --tracker botsort"
echo ""
echo "Доступные трекеры:"
echo "  botsort"
echo "  bytetrack"
echo "  ocsort"
echo "  deepocsort"
echo "  strongsort"
echo "  hybridsort"
echo "  occluboost"
echo "  boosttrack"
echo "  sfsort"
echo "  simple"
echo ""
echo "Если нужно скачать модель заранее:"
echo "  DOWNLOAD_MODEL=1 bash install_cuda128.sh"
echo ""
echo "Если хочешь установить свежий boxmot вместо зафиксированной версии:"
echo "  BOXMOT_PACKAGE=boxmot bash install_cuda128.sh"
echo ""
echo "Если нужна другая CUDA-версия:"
echo "  CUDA_VARIANT=cu126 bash install_cuda128.sh"
echo "  CUDA_VARIANT=cu124 bash install_cuda128.sh"
echo ""
echo "Если нужна CPU-версия:"
echo "  CPU=1 bash install_cuda128.sh"
echo ""
echo "Если нужна ночная сборка PyTorch для CUDA 12.8:"
echo "  PYTORCH_NIGHTLY=1 bash install_cuda128.sh"
