#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/train_with_dataset.sh \
    --train-images /abs/path/train/images \
    --train-json /abs/path/train/annotations.json \
    --test-images /abs/path/test/images \
    --test-json /abs/path/test/annotations.json \
    [--config-dir ./configs/CoStudent_fcos.res50.redcup.few.og] \
    [--train-name coco_cli_train] \
    [--test-name coco_cli_test] \
    [--cuda-visible-devices 0] \
    [--dist-url tcp://127.0.0.1:65454] \
    [-- python tools/train_net.py extra opts...]

Notes:
  - The script registers dataset routes through COSTUDENT_DATASET_OVERRIDES.
  - Paths can be absolute, or relative to DATA_ROOT (defaults to ./datasets).
  - Any arguments after `--` are appended to train_net.py.
EOF
}

TRAIN_IMAGES=""
TRAIN_JSON=""
TEST_IMAGES=""
TEST_JSON=""
CONFIG_DIR="./configs/CoStudent_fcos.res50.redcup.few.og"
TRAIN_NAME="coco_cli_train"
TEST_NAME="coco_cli_test"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES:-0}"
DIST_URL="tcp://127.0.0.1:65454"
DATA_ROOT="${DATA_ROOT:-./datasets}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-images) TRAIN_IMAGES="$2"; shift 2 ;;
    --train-json) TRAIN_JSON="$2"; shift 2 ;;
    --test-images) TEST_IMAGES="$2"; shift 2 ;;
    --test-json) TEST_JSON="$2"; shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --train-name) TRAIN_NAME="$2"; shift 2 ;;
    --test-name) TEST_NAME="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_ARG="$2"; shift 2 ;;
    --dist-url) DIST_URL="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TRAIN_IMAGES" || -z "$TRAIN_JSON" || -z "$TEST_IMAGES" || -z "$TEST_JSON" ]]; then
  echo "Missing required dataset path arguments." >&2
  usage
  exit 2
fi

# Resolve to absolute paths when they are relative.
resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf "%s" "$p"
  else
    printf "%s/%s" "$DATA_ROOT" "$p"
  fi
}

TRAIN_IMAGES_RESOLVED="$(resolve_path "$TRAIN_IMAGES")"
TRAIN_JSON_RESOLVED="$(resolve_path "$TRAIN_JSON")"
TEST_IMAGES_RESOLVED="$(resolve_path "$TEST_IMAGES")"
TEST_JSON_RESOLVED="$(resolve_path "$TEST_JSON")"

export COSTUDENT_DATASET_OVERRIDES
COSTUDENT_DATASET_OVERRIDES="$(cat <<EOF
{
  "${TRAIN_NAME}": ["${TRAIN_IMAGES_RESOLVED}", "${TRAIN_JSON_RESOLVED}"],
  "${TEST_NAME}": ["${TEST_IMAGES_RESOLVED}", "${TEST_JSON_RESOLVED}"]
}
EOF
)"

echo "Using config dir: ${CONFIG_DIR}"
echo "Training dataset: ${TRAIN_NAME}"
echo "  image_root: ${TRAIN_IMAGES_RESOLVED}"
echo "  json_file : ${TRAIN_JSON_RESOLVED}"
echo "Testing dataset : ${TEST_NAME}"
echo "  image_root: ${TEST_IMAGES_RESOLVED}"
echo "  json_file : ${TEST_JSON_RESOLVED}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_ARG}" python tools/train_net.py \
  --dir "${CONFIG_DIR}" \
  --dist-url "${DIST_URL}" \
  DATASETS.TRAIN "('${TRAIN_NAME}',)" \
  DATASETS.TEST "('${TEST_NAME}',)" \
  "${EXTRA_ARGS[@]}"
