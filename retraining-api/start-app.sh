#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

IMAGE="leskovecg/smart-energy-api:1.3.0"
NAME="retraining-api"
HOST_PORT="5004"
CONTAINER_PORT="8000"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

# Ensure local data directories exist (base CSV should already be in data/base/)
mkdir -p "$SCRIPT_DIR/data/base" "$SCRIPT_DIR/data/appended"

echo "[1/4] Pulling image: $IMAGE"
sudo docker pull "$IMAGE"

echo "[2/4] Stopping old container (if exists): $NAME"
sudo docker rm -f "$NAME" 2>/dev/null || true

echo "[3/4] Starting container: $NAME"
sudo docker run -d \
  --name "$NAME" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  --env-file "$ENV_FILE" \
  -v "$SCRIPT_DIR/data:/app/data" \
  --restart unless-stopped \
  "$IMAGE"

echo "[4/4] Health check"
sleep 2
curl -fsS "http://localhost:${HOST_PORT}/health" && echo
echo "Container '$NAME' is up on port ${HOST_PORT}."
