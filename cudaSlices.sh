#!/bin/bash
# Równoległe przeszukiwanie całej przestrzeni grafów z użyciem geng res/mod.
# Użycie:
#   ./cuda_slices.sh <mod> <start_res> [liczba_rdzeni]
#   mod         - na ile równych części dzielimy całą przestrzeń (np. 1000)
#   start_res   - numer części, od której zacząć (0 .. mod-1)
#   liczba_rdzeni - ile równoległych procesów uruchomić (domyślnie wszystkie rdzenie CPU)

if [ $# -lt 2 ]; then
    echo "Użycie: $0 <mod> <start_res> [liczba_rdzeni]"
    exit 1
fi

MOD=$1
START_RES=$2
CORES=${3:-$(nproc)}

N_VERTEX=6
K=10
GENG="../nauty2_8_9/geng"
COMPLG="../nauty2_8_9/complg"
AGCUDA="./agcuda2"

OUTDIR="slices_out"
mkdir -p "$OUTDIR"

echo "=== Start: mod=$MOD, start_res=$START_RES, rdzenie=$CORES ===" >&2
echo "Czas rozpoczęcia: $(date)" >&2

active=0
for ((res=START_RES; res<MOD; res++)); do
    # Jeśli plik wynikowy już istnieje, pomiń tę część
    if [ -f "${OUTDIR}/found_${res}.g6" ]; then
        echo "Część $res/$MOD już gotowa – pomijam." >&2
        continue
    fi

    # Przetwarzanie pojedynczej części w tle
    (
        echo "[$(date +%H:%M:%S)] Start res=$res/$MOD" >&2
        t0=$(date +%s.%N)

        ${GENG} -q -c $N_VERTEX ${K}:${K} ${res}/${MOD} 2>/dev/null \
            | ${COMPLG} -q 2>/dev/null \
            | ${AGCUDA} > "${OUTDIR}/found_${res}.g6" 2> "${OUTDIR}/time_${res}.log"

        t1=$(date +%s.%N)
        # Poprawa formatowania, aby uniknąć ".123" bez wiodącego zera
        wall_time=$(echo "$t1 - $t0" | bc -l | sed 's/^\./0./')

        # Czekamy chwilę, aż plik czasu zostanie zapisany (maks. 2 sekundy)
        for try in $(seq 1 20); do
            if [ -s "${OUTDIR}/time_${res}.log" ]; then
                break
            fi
            sleep 0.1
        done

        if [ -s "${OUTDIR}/time_${res}.log" ]; then
            agcuda_time=$(grep -oP '[\d.]+' "${OUTDIR}/time_${res}.log" 2>/dev/null | head -1)
            agcuda_time=${agcuda_time:-0}
        else
            agcuda_time="0"
        fi

        echo "[$(date +%H:%M:%S)] Koniec res=$res/$MOD. Wall: ${wall_time}s, AGCUDA: ${agcuda_time}s" >&2
    ) &

    active=$((active + 1))
    if [ $active -ge $CORES ]; then
        wait -n               # czekamy na zakończenie jednego procesu
        active=$((active - 1))
    fi
done

wait
echo "=== Wszystkie części od $START_RES do $((MOD-1)) zakończone. ===" >&2
echo "Czas zakończenia: $(date)" >&2