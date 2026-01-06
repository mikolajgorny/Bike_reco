#  Bike Model Recognizer - Praca Inżynierska

Projekt i realizacja systemu rozpoznawania modeli rowerów szosowych na zdjęciach. Aplikacja wykorzystuje głębokie uczenie (**ResNet50**) oraz detekcję obiektów (**YOLOv8**) do identyfikacji 12 modeli rowerów od producentów: **Trek, Canyon, Specialized oraz Cervélo**.

**Autor:** Mikołaj Górny  
**Uczelnia:** Politechnika Warszawska, Wydział Elektroniki i Technik Informacyjnych

---

## 🚀 Funkcjonalności

* **Rozpoznawanie modelu:** Klasyfikacja zdjęcia z wykorzystaniem sieci neuronowych.
* **Przeglądarka bazy:** Galeria zdjęć treningowych dla poszczególnych marek.
* **Quiz:** Gra edukacyjna "Zgadnij, co to za rower".
* **Forum:** Prototyp modułu społecznościowego.

---

##  Instrukcja uruchomienia (Krok po kroku)

Aby uruchomić aplikację na własnym komputerze, wykonaj poniższe kroki.

### 1. Pobranie projektu
Pobierz repozytorium na dysk (klonując gitem lub pobierając jako ZIP) i wejdź do folderu:

git clone [https://github.com/mikolajgorny/Bike_reco.git](https://github.com/mikolajgorny/Bike_reco.git)
cd Bike_reco

### 2. Przygotowanie środowiska
Zaleca się stworzenie wirtualnego środowiska Python (venv), aby uniknąć konfliktów bibliotek.

MACOS/LINUX:

python3 -m venv venv
source venv/bin/activate

WINDOWS:

python -m venv venv
venv\Scripts\activate

### 3. Instalacja bibliotek 
Zainstaluj wymagane zależności zapisane w pliku requirements.txt:

pip install -r requirements.txt

### 4. Weryfikacja struktury plików
Przed uruchomieniem upewnij się, że w folderze projektu znajdują się kluczowe pliki i foldery (szczególnie model i logotypy):

Bike_reco/
├── app_final.py                       # Główny plik aplikacji
├── requirements.txt                   # Lista bibliotek
├── yolov8n.pt                         # Model YOLO (pobierze się automatycznie przy starcie)
├── models/
│   └── unified/
│       └── resnet50_12class_transfer.pth  # <--- GŁÓWNY MODEL (Wymagany!)
├── dataset/                           # Folder ze zdjęciami (wymagany do Quizu)
└── logo/                              # Folder z logotypami marek

### 5. Uruchomienie aplikacji
Gdy wszystko jest gotowe, wpisz w terminalu polecenie:

streamlit run app_final.py

Aplikacja otworzy się automatycznie w Twojej domyślnej przeglądarce pod adresem http://localhost:8501.

