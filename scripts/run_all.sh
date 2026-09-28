#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for week in week*.sh; do
    bash "$week"
done
