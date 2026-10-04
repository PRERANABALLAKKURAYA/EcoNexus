#!/usr/bin/env bash
# Usage: REGISTRY=registry.example.com/team ./scripts/push_and_deploy.sh econexus 42
# Registry login is done beforehand by Jenkins (credentials store); nothing is hard-coded here.
set -euo pipefail
IMAGE="${1:?image name}"; TAG="${2:?tag}"
: "${REGISTRY:?set REGISTRY, e.g. registry.example.com/team}"
FULL="$REGISTRY/$IMAGE:$TAG"
docker tag "$IMAGE:$TAG" "$FULL"
docker push "$FULL"
sed "s#<registry>/econexus:<tag>#$FULL#" k8s/deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/econexus --timeout=120s
echo "Rollback if needed: kubectl rollout undo deployment/econexus"
