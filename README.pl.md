# Wersje językowe / Language versions

- [English](README.md)
- [Polski](README.pl.md)

# Spójne grafy całkowite rzędu n = 16 i o ilości krawędzi k = 87

## 1. Streszczenie

Celem projektu jest implementacja algorytmu selekcji [grafów całkowitych](https://en.wikipedia.org/wiki/Integral_graph) (ang. *integral graphs*) oraz analiza przyspieszenia uzyskanego przez zastosowanie technik równoległych. Rozważany problem dotyczy spójnych grafów prostych o zadanej liczbie wierzchołków n=16 i krawędzi k=87, dla każdego grafu wyznaczane są wartości własne jego macierzy sąsiedztwa, a następnie badana jest ich całkowitość z zadaną tolerancją numeryczną. Tylko te grafy, których widmo składa się wyłącznie z liczb całkowitych, zostają wypisane na standardowe wyjście.

Projekt obejmuje trzy warianty obliczeniowe: AGS (program sekwencyjny, stanowiący punkt odniesienia dla pomiarów), AGOMP (wersja zrównoleglona na procesorze przy użyciu OpenMP), AGCUDA (implementacja akcelerowana z użyciem karty graficznej i technologii CUDA).

Głównym celem pracy jest wykazanie, w jakim stopniu kolejne poziomy równoległości (wielordzeniowy CPU, masywnie równoległe GPU) skracają czas filtrowania grafów oraz identyfikacja ograniczeń każdej z architektur, m.in.: narzutu na komunikację, rozmiaru zadań oraz skalowalności.

## 2. Opis problemu

[Graf całkowity](https://en.wikipedia.org/wiki/Integral_graph) (ang. *integral graph*) to graf prosty, którego wszystkie wartości własne macierzy sąsiedztwa są liczbami całkowitymi. Wyznaczanie i klasyfikacja takich grafów dla ustalonego rzędu *n* i liczby krawędzi *k* stanowi otwarty problem kombinatoryczny o znaczeniu zarówno teoretycznym, jak i praktycznym, czyli widma grafów całkowitych znajdują zastosowanie m.in. w teorii informacji kwantowej oraz przy konstrukcji sieci o zadanych właściwościach spektralnych.

W ramach projektu badane są spójne grafy o *n=16* wierzchołkach i *k=87* krawędziach. Przestrzeń poszukiwań jest bardzo duża, a odsetek grafów całkowitych w tym zbiorze jest niewielki, co wymusza zastosowanie wydajnych metod obliczeniowych. Generowaniem wszystkich nieizomorficznych grafów spełniających zadane parametry zajmuje się zewnętrzny program *geng* z pakietu *nauty*, który wypisuje grafy w formacie *graph6* na standardowe wyjście. Strumień ten stanowi wejście dla programu filtrującego *AGS* (w wersji sekwencyjnej) lub jego odpowiedników równoległych, który dla każdego grafu wyznacza wartości własne macierzy sąsiedztwa, a następnie sprawdza ich całkowitość. Wyłącznie grafy o widmie całkowitym są przepuszczane na wyjście.

Z punktu widzenia obliczeń numerycznych problem sprowadza się do zadania wyznaczania widma symetrycznej macierzy rzeczywistej i weryfikacji, czy każda z wartości własnych jest w granicy tolerancji numerycznej liczbą całkowitą. Zastosowana metoda bazuje na redukcji Householdera do postaci trójdiagonalnej, po czym przez bisekcję wykorzystującą sekwencje Sturma izolowane są kolejne wartości własne.

Wybór trzech wariantów algorytmu (sekwencyjnego, równoległego CPU oraz równoległego GPU) umożliwia ocenę, w jakim stopniu różne architektury sprzętowe mogą przyspieszyć filtrowanie dużych zbiorów grafów.

## 3. Spis zaimplementowanych algorytmów

| Lp. | Algorytm (KOD) | Kategoria | Przeznaczenie | Uwagi |
|-----|----------------|-----------|---------------|-------|
| 1 | GEG | Generowanie danych | Tworzenie grafów nieskierowanych o zadanej liczbie wierzchołków i krawędzi | Wykorzystuje pakiet nauty |
| 2 | AGS | Sekwencyjny | Sekwencyjne wyszukiwanie grafów całkowitych | Bazuje na sicie spektralnym |
| 3 | AGOMP | Równoległy (CPU) | Równoległe wyszukiwanie grafów całkowitych z wykorzystaniem wielu wątków | Wykorzystuje OpenMP |
| 4 | AGCUDA | Równoległy (GPU) | Równoległe wyszukiwanie grafów całkowitych z wykorzystaniem karty graficznej | Wykorzystuje CUDA |

## 4. Zaimplementowane algorytmy

### 4.1. GEG - Generowanie danych oraz program pomiarowy

#### 4.1.1. geng

Do generowania grafów wejściowych wykorzystywane jest narzędzie `geng` wchodzące w skład pakietu `nauty`. Wywołanie

<code> geng -c 16 87:87 </code>

produkuje wszystkie nieizomorficzne grafy spójne o 16 wierzchołkach i 87 krawędziach, wypisując każdy z nich w jednym wierszu w formacie `graph6`. Opcja `-c` wymusza spójność, a parametry liczbowe określają odpowiednio rząd grafu oraz przedział liczby krawędzi. Ponieważ celem projektu jest również zbadanie skalowania algorytmu wraz z rosnącą wielkością danych, przygotowano stały zbiór referencyjny zawierający dziesięć milionów wygenerowanych grafów:

<code> geng -c 16 87:87 | head -n 10000000 > graphSet.g6 </code>

#### 4.1.2. Program pomiarowy

zawartość computeTests

### 4.2. AGS - Rozwiązanie sekwencyjne

#### 4.2.1. Opis

Jako podstawa programu AGS zostało użyte `sito8.cu`. Została usunięta cała infrastruktura związana z CUDA – alokowanie pamięci na GPU, kopiowanie tam danych wejściowych i synchronizacja po wykonaniu jądra. Zamiast uruchamiania jądra na karcie, obliczenia przeniesiono do zwykłej funkcji C, która operuje bezpośrednio na buforze w pamięci procesora.

Warto podkreślić, że pierwotny kod CUDA używał tylko jednego bloku i jednego wątku, czyli jądro wykonywało się więc tak samo sekwencyjnie, jak zwykły program na CPU. Usunięcie szkieletu GPU nie zmieniło więc stopnia zrównoleglenia, bo go tam nigdy nie było.

Dzięki tej zmianie program AGS nie ponosi już opóźnień związanych z kopiowaniem danych i startowaniem jądra na karcie. Daje to czysty, referencyjny wariant, który będzie naturalnym punktem odniesienia przy ocenie przyspieszenia wersji wykorzystujących OpenMP i wielowątkowe CUDA.

W celu precyzyjnego pomiaru czasu przetwarzania, bez zanieczyszczenia wyników operacjami wejścia/wyjścia, do kodu wstawiono znaczniki czasu wykorzystujące funkcję `omp_get_wtime()`. Mierzą one tylko pętlę główną przetwarzającą kolejne grafy. Czas wypisywany jest na standardowe wyjście błędów.

#### 4.2.2. Kompilacja i uruchomienie

Kompilacja:

<code> gcc -fopenmp -o ags ags.c -lm </code>

Uruchomienie dla zbioru danych z przekierowaniem do plików wynikowych:

<code> ./ags < input.g6 > output.g6 2> czas.log </code>

Uruchomienie z geng:

<code> ./nauty2_8_9/geng -c 16 87:87 2>/dev/null | ./ags </code>

### 4.3. AGOMP - OpenMP

#### 4.3.1. Opis

W programie AGOMP zrównoleglenie przetwarzania grafów uzyskano przez zastosowanie dyrektyw OpenMP w modelu zadań. Wątek główny odczytuje kolejne wiersze ze standardowego wejścia i gromadzi je w lokalnej tablicy o stałej pojemności (domyślnie 1024 elementy). Gdy tablica zostaje zapełniona, cała paczka przekazywana jest do osobnego zadania (task), które wykonuje się asynchronicznie na dostępnych wątkach. Zadanie to sekwencyjnie wywołuje funkcję analizującą dla każdego z wierszy w paczce, a po zakończeniu przetwarzania zwalnia pamięć.

Dzięki temu wątek główny może natychmiast kontynuować odczyt i przygotowywanie kolejnej partii danych, podczas gdy pozostałe wątki zajmują się obliczeniami. Po wyczerpaniu strumienia zgłaszane jest ostatnie, niepełne zadanie, a program synchronizuje się ze wszystkimi zadaniami. Wypisywanie wyników na standardowe wyjście chronione jest sekcją krytyczną, co zapewnia atomowość komunikatu.

Pomiar czasu obejmuje cały okres działania, od rozpoczęcia odczytu pierwszego grafu do zakończenia przetwarzania ostatniego wiersza. Wynik trafia na stderr.

#### 4.3.2. Kompilacja i uruchomienie

Kompilacja:

<code> gcc -fopenmp -o agomp agomp.c -lm </code>

Uruchomienie bez jawnego podawania liczby wątków:

<code> ./agomp < input.g6 > output.g6 2> czas.log </code>

Uruchomienie z jawnym podawaniem liczby wątków `N`

<code> ./agomp N < input.g6 > output.g6 2> czas.log </code>

Uruchomienie z geng:

<code> ./geng -c 16 87:87 2>/dev/null | ./agomp </code>

### 4.4. AGCUDA - CUDA

#### 4.4.1. Opis

*Sekcja nie jest jeszcze dostępna. Implementacja wersji CUDA jest w trakcie przygotowywania.*

#### 4.4.2. Kompilacja i uruchomienie

*Sekcja nie jest jeszcze dostępna.*

## 5. Analiza

### 5.1. Analiza dla rosnącej liczby grafów

Zbadano skalowalność implementacji sekwencyjnej (AGS) oraz równoległej na procesorze (AGOMP) w funkcji liczby przetworzonych grafów. Pomiary przeprowadzono dla 14 rozmiarów próbki `N`, od 1000 do 8192000 wierszy, generowanych jako kolejne potęgi dwójki pomnożone przez 1000. Dla każdego `N` oba programy uruchomiono pięciokrotnie; w tabeli zamieszczono minimalny uzyskany czas. Program AGOMP uruchamiano z liczbą wątków równą liczbie dostępnych rdzeni, w tym przypadku jest to 4. Dane dla wersji akcelerowanej graficznie (AGCUDA) nie są jeszcze dostępne.

**Tabela 1. Czasy obliczeń programów AGS, AGOMP oraz AGCUDA wraz z przyspieszeniami dla rosnącej liczby grafów wejściowych. S - przyspieszenie**

| N | AGS [s] | AGOMP [s] | AGCUDA [s] | S_AGOMP | S_AGCUDA |
|---|---------|-----------|------------|---------|----------|
| 1000 | 0.018146 | 0.016122 | - | 1.13 | - |
| 2000 | 0.032739 | 0.022493 | - | 1.46 | - |
| 4000 | 0.065713 | 0.043216 | - | 1.52 | - |
| 8000 | 0.125000 | 0.076020 | - | 1.64 | - |
| 16000 | 0.259339 | 0.153810 | - | 1.69 | - |
| 32000 | 0.472716 | 0.283684 | - | 1.67 | - |
| 64000 | 1.008297 | 0.556530 | - | 1.81 | - |
| 128000 | 2.045158 | 1.087607 | - | 1.88 | - |
| 256000 | 3.973273 | 2.070440 | - | 1.92 | - |
| 512000 | 8.405879 | 3.910184 | - | 2.15 | - |
| 1024000 | 15.896943 | 8.307194 | - | 1.91 | - |
| 2048000 | 31.331593 | 17.107519 | - | 1.83 | - |
| 4096000 | 67.700554 | 34.379473 | - | 1.97 | - |
| 8192000 | 140.127044 | 68.934237 | - | 2.03 | - |

Zebrane wyniki wskazują, że program AGS skaluje się w sposób niemal liniowy, każdorazowe podwojenie liczby grafów powoduje proporcjonalny przyrost czasu obliczeń. Taki przebieg jest spodziewany, ponieważ algorytm analizuje każdy graf osobno, a rozkład kosztu na pojedynczy graf jest zbliżony.

Wyniki uzyskane dla programu AGOMP wskazują na umiarkowany, lecz zauważalny zysk z równoległości. Przyspieszenie względem wersji sekwencyjnej rośnie stopniowo od 1,13× dla 1000 grafów do wartości oscylujących wokół 2× dla największych badanych zbiorów (4–8 milionów wierszy). Oznacza to, że zastosowany model współbieżnego przetwarzania strumieniowego pozwala skrócić czas obliczeń mniej więcej dwukrotnie przy wykorzystaniu ośmiu wątków sprzętowych.

Ograniczone skalowanie wynika przede wszystkim z obecności nieuniknionej części sekwencyjnej, na którą składają się odczyt danych ze standardowego wejścia oraz operowanie na współdzielonych buforach. O ile sam algorytm analizy grafu został w pełni zrównoleglony, o tyle faza wejścia pozostaje sekwencyjna i stanowi coraz większy udział w całkowitym czasie przy mniejszych próbkach. Dodatkowo, przy większej liczbie wątków narasta efekt nasycenia magistrali pamięci, ponieważ wszystkie jednostki wykonawcze odwołują się do wspólnych obszarów danych. W rezultacie maksymalne przyspieszenie nie przekracza dwukrotności i stabilizuje się na tym poziomie niezależnie od dalszego wzrostu rozmiaru zadania.

### 5.2. Analiza wpływu liczby wątków na wydajność AGOMP

W celu zbadania skalowalności wątków w programie AGOMP wykonano pomiary czasu dla czterech rozmiarów danych: 1 tys., 10 tys., 100 tys. oraz 1 mln grafów. Dla każdej kombinacji wielkości zadania i liczby wątków (od 1 do 10) przeprowadzono pięć powtórzeń, a w tabeli zestawiono czasy minimalne. Jako punkt odniesienia do obliczenia przyspieszenia przyjęto czas uzyskany przy jednym wątku (S = T₁ / Tₙ). Procesor na którym wykonano pomiary posiada 4 rdzenie x 2 wątki.

**Tabela 2. Czasy wykonania AGOMP [s] dla różnych liczb wątków oraz przyspieszenie**

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | S₂ | S₃ | S₄ | S₅ | S₆ | S₇ | S₈ | S₉ | S₁₀ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.014746 | 0.015136 | 0.014849 | 0.015166 | 0.016572 | 0.017878 | 0.017634 | 0.017180 | 0.017396 | 0.016250 | 0.97 | 0.99 | 0.97 | 0.89 | 0.82 | 0.84 | 0.86 | 0.85 | 0.91 |
| 10000 | 0.156311 | 0.080764 | 0.080860 | 0.078457 | 0.075567 | 0.085501 | 0.083592 | 0.090033 | 0.090252 | 0.090395 | 1.94 | 1.93 | 1.99 | 2.07 | 1.83 | 1.87 | 1.74 | 1.73 | 1.73 |
| 100000 | 1.446329 | 0.802272 | 0.796545 | 0.777778 | 0.797517 | 0.770019 | 0.655603 | 0.815391 | 0.812700 | 0.635903 | 1.80 | 1.82 | 1.86 | 1.81 | 1.88 | 2.21 | 1.77 | 1.78 | 2.27 |
| 1000000 | 16.406792 | 8.230509 | 8.289438 | 8.413557 | 8.629064 | 9.633345 | 9.387560 | 9.566058 | 9.505468 | 9.302202 | 1.99 | 1.98 | 1.95 | 1.90 | 1.70 | 1.75 | 1.71 | 1.73 | 1.76 |

Dla zbioru 1000 grafów czas przetwarzania praktycznie nie zależy od liczby wątków, a w większości przypadków jest nawet nieznacznie dłuższy niż przy jednym wątku. Przy tak małym zadaniu narzut wynikający z tworzenia paczek i synchronizacji zadań OpenMP przewyższa potencjalny zysk z równoległości.

**Tabela 3. Liczba przetwarzanych grafów na sekundę AGOMP**

| N | Najlepszy czas [s] | Grafy/s |
|---|--------------------|---------|
| 1000 | 0.014746 | 67 815 |
| 10000 | 0.075567 | 132 333 |
| 100000 | 0.635903 | 157 257 |
| 1000000 | 8.230509 | 121 499 |

Najwyższą przepustowość osiągnięto dla zbioru 100 000 grafów – około 157 tysięcy grafów na sekundę. Dla największego zbioru przepustowość nieznacznie spada, co może być spowodowane efektami cache'owania i zarządzaniem pamięcią.

### 5.3. Zestawienie poprawności działania

*Sekcja nie jest jeszcze dostępna. Porównanie wyników działania AGS, AGOMP i AGCUDA z weryfikacją poprawności zostanie uzupełnione po ukończeniu implementacji CUDA.*
