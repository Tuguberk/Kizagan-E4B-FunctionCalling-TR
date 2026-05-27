#!/usr/bin/env bash
# ============================================================
# Turkish LLM Tool-Calling Benchmark — Tam Çalıştırma Scripti
#
# Kullanım:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh
#
# Adımlar:
#   1. Bağımlılıkları kur
#   2. Base modeli değerlendir → results/base.json
#   3. Fine-tuned modeli değerlendir → results/finetuned.json
#   4. İki modeli karşılaştır
# ============================================================

set -euo pipefail

BASE_MODEL="AlicanKiraz0/Kizagan-E4B-Turkish-Reasoning-Model"
FT_MODEL="Tuguberk/Kizagan-E4B-FunctionCalling-TR"
QUESTIONS="questions.json"
RESULTS_DIR="results"

# Renk kodları
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[benchmark]${NC} $*"; }
ok()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn(){ echo -e "${YELLOW}[!]${NC} $*"; }

# ─── 1. Bağımlılıklar ────────────────────────────────────────────────────────
log "Bağımlılıklar kontrol ediliyor..."
pip install --quiet transformers torch accelerate tqdm huggingface_hub
ok "Bağımlılıklar hazır."

# ─── 2. Base model ───────────────────────────────────────────────────────────
log "Base model değerlendiriliyor: ${BASE_MODEL}"
python evaluate.py \
    --model   "${BASE_MODEL}" \
    --questions "${QUESTIONS}" \
    --output  "${RESULTS_DIR}/base.json" \
    --device  auto

ok "Base model tamamlandı → ${RESULTS_DIR}/base.json"

# ─── 3. Fine-tuned model ─────────────────────────────────────────────────────
log "Fine-tuned model değerlendiriliyor: ${FT_MODEL}"
python evaluate.py \
    --model   "${FT_MODEL}" \
    --questions "${QUESTIONS}" \
    --output  "${RESULTS_DIR}/finetuned.json" \
    --device  auto

ok "Fine-tuned model tamamlandı → ${RESULTS_DIR}/finetuned.json"

# ─── 4. Karşılaştırma ────────────────────────────────────────────────────────
log "Karşılaştırma raporu:"
python evaluate.py --compare "${RESULTS_DIR}/base.json" "${RESULTS_DIR}/finetuned.json"

ok "Benchmark tamamlandı!"
echo ""
echo "Sonuç dosyaları:"
echo "  ${RESULTS_DIR}/base.json"
echo "  ${RESULTS_DIR}/finetuned.json"
