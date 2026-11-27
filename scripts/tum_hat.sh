#!/usr/bin/env bash
# Tüm sonuç hattını sırayla çalıştırır: deney -> nihai model -> grafikler -> rapor -> testler.
#   bash scripts/tum_hat.sh            # günlükler outputs/*_log.txt, durum outputs/HAT_DURUM.txt
#   KOMIT=1 bash scripts/tum_hat.sh    # sonunda yerel git commit de yapar (push yapmaz)
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0          # başka işlerle paylaşılan CPU'da iş parçacığı çekişmesini önler
export PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p outputs
DURUM=outputs/HAT_DURUM.txt
adim() { echo "$(date '+%F %T') $*" | tee -a "$DURUM"; }

: > "$DURUM"
adim "başladı"
$PY scripts/deney.py ${DENEY_ARG:-} > outputs/deney_log.txt 2>&1 || { adim "HATA: deney.py (outputs/deney_log.txt)"; exit 1; }
adim "deney.py bitti"
$PY scripts/egit_final.py ${FINAL_ARG:-} > outputs/final_log.txt 2>&1 || { adim "HATA: egit_final.py (outputs/final_log.txt)"; exit 1; }
adim "egit_final.py bitti"
$PY scripts/gorseller.py > outputs/gorsel_log.txt 2>&1 || { adim "HATA: gorseller.py"; exit 1; }
$PY scripts/rapor.py >> outputs/gorsel_log.txt 2>&1 || { adim "HATA: rapor.py"; exit 1; }
adim "grafikler ve rapor bitti"
if $PY -m pytest > outputs/pytest_log.txt 2>&1; then TEST="testler geçti"; else TEST="TESTLER BAŞARISIZ (outputs/pytest_log.txt)"; fi
adim "$TEST: $(tail -1 outputs/pytest_log.txt)"

if [ "${KOMIT:-0}" = "1" ]; then
  git add -A
  git commit -q -F - <<EOF && adim "commit: $(git log --oneline -1)" || adim "commit yapılamadı"
Gerçek eğitim sonuçlarını, nihai modeli ve grafikleri üret

- results/metrics.json: 4 model x 4 deney (scripts/deney.py)
- results/demo_trend.json ve grafikler; docs/SONUCLAR.md otomatik üretildi
- Test durumu: $TEST

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
fi
if [ "${YAYINLA:-0}" = "1" ]; then
  if [ "$TEST" != "testler geçti" ]; then adim "YAYIN ATLANDI: testler geçmedi"; adim "BİTTİ"; exit 1; fi
  DAL=$(git rev-parse --abbrev-ref HEAD)
  git push -u origin "$DAL" >> outputs/yayin_log.txt 2>&1 && adim "push: $DAL" || { adim "HATA: push"; exit 1; }
  if ! gh pr view "$DAL" --json number >/dev/null 2>&1; then
    gh pr create --base main --head "$DAL" --title "${PR_BASLIK:-Polar NLP v0.3: yeniden yapılandırma, dürüst değerlendirme, genişletilmiş veri}"       --body-file "${PR_GOVDE:-docs/SONUCLAR.md}" >> outputs/yayin_log.txt 2>&1 && adim "PR açıldı: $(gh pr view "$DAL" --json url -q .url)" || { adim "HATA: PR açılamadı"; exit 1; }
  fi
  gh pr merge "$DAL" --merge >> outputs/yayin_log.txt 2>&1 && adim "PR birleştirildi" || { adim "HATA: PR birleştirilemedi (outputs/yayin_log.txt)"; exit 1; }
fi
adim "BİTTİ"
