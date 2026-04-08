#!/bin/bash
set -e

echo "==> 1/4 Stopping and removing orphaned containers..."
docker rm -f xplagiax-api xplagiax-worker redis qdrant 2>/dev/null || true
docker network rm xplagiax-net 2>/dev/null || true
docker network create xplagiax-net

echo "==> 2/4 Purging cache and rebuilding image..."
docker build --no-cache -t xplagiax:latest .

echo "==> 3/4 Starting dependencies (Redis, Qdrant)..."
docker run -d --name redis --network xplagiax-net -p 6379:6379 redis:alpine
docker run -d --name qdrant --network xplagiax-net -p 6333:6333 qdrant/qdrant:latest

# Wait for dependencies
sleep 5

echo "==> 4/4 Starting API and Worker..."
docker run -d --name xplagiax-api \
  --network xplagiax-net \
  -p 5004:5004 \
  -e REDIS_HOST=redis \
  -e QDRANT_HOST=qdrant \
  xplagiax:latest api

docker run -d --name xplagiax-worker \
  --network xplagiax-net \
  -e REDIS_HOST=redis \
  -e QDRANT_HOST=qdrant \
  xplagiax:latest worker

echo "==> Done! Services are up."
