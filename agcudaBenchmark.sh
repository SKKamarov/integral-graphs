#!/bin/bash

echo "Uruchamiam 4 procesy na 60s..." >&2

# Terminal 1
timeout 60s bash -c '../nauty2_8_9/geng -q -c 16 87:87 -s0 -e24 2>/dev/null | ../nauty2_8_9/complg -q 2>/dev/null | ./agcuda2 > /dev/null 2> czas_0.log' &
PID0=$!

# Terminal 2
timeout 60s bash -c '../nauty2_8_9/geng -q -c 16 87:87 -s25 -e49 2>/dev/null | ../nauty2_8_9/complg -q 2>/dev/null | ./agcuda2 > /dev/null 2> czas_1.log' &
PID1=$!

# Terminal 3
timeout 60s bash -c '../nauty2_8_9/geng -q -c 16 87:87 -s50 -e74 2>/dev/null | ../nauty2_8_9/complg -q 2>/dev/null | ./agcuda2 > /dev/null 2> czas_2.log' &
PID2=$!

# Terminal 4
timeout 60s bash -c '../nauty2_8_9/geng -q -c 16 87:87 -s75 -e99 2>/dev/null | ../nauty2_8_9/complg -q 2>/dev/null | ./agcuda2 > /dev/null 2> czas_3.log' &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3

# Wyciągnij czasy
T0=$(grep -oP '[\d.]+' czas_0.log | tail -1)
T1=$(grep -oP '[\d.]+' czas_1.log | tail -1)
T2=$(grep -oP '[\d.]+' czas_2.log | tail -1)
T3=$(grep -oP '[\d.]+' czas_3.log | tail -1)

echo "Czasy: $T0 $T1 $T2 $T3" >&2

# Suma czasów
SUM=$(echo "$T0 + $T1 + $T2 + $T3" | bc -l)
echo "Sumaryczny czas GPU: ${SUM}s" >&2

# Przepustowość – załóż że każdy proces przetworzył tyle samo grafów.
# Jeśli chcesz dokładnie, dołóż `pv` jak wcześniej.
echo "Aby obliczyć przepustowość, podziel liczbę grafów (np. 10M) przez ${SUM}" >&2