import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import os
import random
from ultralytics import YOLO
from PIL import Image

# ======================
# KONFIGURACJA
# ======================

st.set_page_config(page_title="Bike Model Recognizer", page_icon=":bike:", layout="centered")

st.markdown("<h1 style='text-align: center;'>🚴 Rozpoznawanie modelu roweru</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Wczytaj zdjęcie swojego roweru, a aplikacja spróbuje odgadnąć markę i model!</p>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>🛠️ Obsługiwane marki:</h3>", unsafe_allow_html=True)

# === Loga 4 marek ===
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("logo/treklogo.png", width=100)
with col2:
    st.image("logo/canyonlogo.png", width=150)
with col3:
    st.image("logo/speclogo.png", width=100)
with col4:
    st.image("logo/Cervelo-Logo.wine.png", width=110)

# ======================
# WCZYTANIE MODELU
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wszystkie klasy (12 modeli)
all_classes = [
    "aeroad", "aethos", "domane", "emonda", "endurance", "madone",
    "r5", "roubaix", "s5", "soloist", "tarmac", "ultimate"
]

model_path = "models/unified/resnet50_12class_transfer.pth"

def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model(model_path, len(all_classes))

# ======================
# TRANSFORMACJE
# ======================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# FUNKCJA PREDYKCJI
# ======================

# Gotowy model YOLO
detector = YOLO("yolov8n.pt")  

def contains_bicycle(image_pil):
    results = detector(image_pil, verbose=False)
    for result in results:
        classes = result.names
        detected = result.boxes.cls.tolist()
        for class_id in detected:
            if classes[int(class_id)] == 'bicycle':
                return True
    return False

def predict_bike(image):
    # Najpierw sprawdź, czy jest rower
    if not contains_bicycle(image):
        return None, None
    # Dopiero wtedy klasyfikuj
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return all_classes[pred.item()], conf.item()

# ======================
# FUNKCJA: WYŚWIETLANIE ROWERÓW
# ======================

def display_bikes(brand):
    base_path = f"dataset/{brand}"
    if not os.path.exists(base_path):
        st.warning("Brak danych dla wybranej marki.")
        return

    models = [m for m in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, m))]
    models = [m for m in models if "augmented" not in m.lower()]

    for model in models:
        st.subheader(f"Model: {model.capitalize()}")
        model_path = os.path.join(base_path, model)
        images = [img for img in os.listdir(model_path) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))][:5]

        if images:
            cols = st.columns(len(images))
            for idx, img_name in enumerate(images):
                try:
                    image = Image.open(os.path.join(model_path, img_name))
                    with cols[idx]:
                        st.image(image, caption=model.capitalize(), use_container_width=True)
                except:
                    st.warning(f"Nie udało się załadować: {img_name}")

# ======================
# QUIZ – LOSOWE ZDJĘCIA
# ======================

def get_random_bike_image(dataset_path="dataset"):
    brands = [b for b in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, b))]
    brand = random.choice(brands)
    models = [m for m in os.listdir(os.path.join(dataset_path, brand)) if os.path.isdir(os.path.join(dataset_path, brand, m))]
    model = random.choice(models)
    image_dir = os.path.join(dataset_path, brand, model)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if not image_files:
        return None, None, None
    image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, image_file)
    return image_path, f"{brand.capitalize()} {model.capitalize()}", brands

# ======================
# INTERFEJS
# ======================

model_classes = {
    'trek': ['domane', 'emonda', 'madone'],
    'canyon': ['aeroad', 'endurance', 'ultimate'],
    'specialized': ['aethos', 'roubaix', 'tarmac'],
    'cervelo': ['r5', 's5', 'soloist']
}

tab1, tab2, tab3, tab5, tab4 = st.tabs([
    "🔍 Sprawdź rower",
    "🚴‍♂️ Przeglądaj rowery",
    "🧠 Zgadnij model",
    "📢 Forum",
    "ℹ️ O projekcie"
])

# Tab 1 – Rozpoznawanie

with tab1:
    st.markdown("<h2 style='text-align: center;'>📷 Wczytaj zdjęcie roweru</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Wczytane zdjęcie", use_container_width=True)
        if st.button("🔍 Sprawdź swój rower"):
            with st.spinner("Analizuję zdjęcie..."):
                model_name, confidence = predict_bike(image)
                if model_name is None:
                    st.warning("🚫 To zdjęcie nie przedstawia roweru lub nie można rozpoznać modelu.")
                else:
                    st.success(f"✅ Rozpoznano model: **{model_name.upper()}** ({confidence * 100:.2f}% pewności)")

# Tab 2 – Przegląd
with tab2:
    st.header("🚴‍♂️ Przeglądaj dostępne modele rowerów")
    brand = st.selectbox("Wybierz markę:", ["Trek", "Canyon", "Specialized", "Cervelo"])
    if brand:
        display_bikes(brand.lower())

# Tab 3 – Quiz
with tab3:
    st.header("🧐 Zgadnij, co to za rower")

    quiz_mode = st.radio("Wybierz tryb:", ["Pojedynczy quiz", "5-rundowa gra"])

    if quiz_mode == "Pojedynczy quiz":
        if "quiz_image" not in st.session_state:
            img_path, correct_label, _ = get_random_bike_image()
            st.session_state.quiz_image = img_path
            st.session_state.correct_answer = correct_label

        if st.button("🔄 Wylosuj inne zdjęcie"):
            img_path, correct_label, _ = get_random_bike_image()
            st.session_state.quiz_image = img_path
            st.session_state.correct_answer = correct_label

        if st.session_state.quiz_image:
            st.image(st.session_state.quiz_image, caption="Zgadnij, jaki to rower!", use_container_width=True)

            all_labels = [
                f"{brand.capitalize()} {model.capitalize()}"
                for brand, models in model_classes.items()
                for model in models
            ]
            answer = st.selectbox("Wybierz markę i model:", all_labels)

            if st.button("✅ Sprawdź odpowiedź"):
                if answer == st.session_state.correct_answer:
                    st.success("✅ Dobrze! To jest dokładnie ten rower!")
                else:
                    st.error(f"❌ Niestety, to był: {st.session_state.correct_answer}")

    elif quiz_mode == "5-rundowa gra":
        if "game_round" not in st.session_state:
            st.session_state.game_round = 1
            st.session_state.game_score = 0
            st.session_state.game_finished = False
            st.session_state.awaiting_next = False

        if "game_image" not in st.session_state or st.session_state.get("refresh_question", False):
            img_path, correct_label, _ = get_random_bike_image()
            st.session_state.game_image = img_path
            st.session_state.game_answer = correct_label
            st.session_state.refresh_question = False
            st.session_state.awaiting_next = False

        if st.session_state.game_finished:
            st.markdown("<h2 style='text-align: center;'>🎉 KONIEC GRY!</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Twój wynik: {st.session_state.game_score}/5</h3>", unsafe_allow_html=True)

            if st.session_state.game_score == 5:
                st.success("🚴‍♂️ Perfekcyjnie! Jesteś mistrzem rozpoznawania rowerów!")
            elif st.session_state.game_score >= 3:
                st.info("👍 Dobry wynik! Spróbuj jeszcze raz i zobacz, czy uda się lepiej!")
            else:
                st.warning("👀 Nie poszło najlepiej – ale głowa do góry, spróbuj jeszcze raz!")

            st.markdown("---")
            center_col = st.columns(3)[1]
            with center_col:
                if st.button("🔁 Zagraj ponownie"):
                    for key in [
                        "game_round", "game_score", "game_finished",
                        "refresh_question", "awaiting_next",
                        "game_image", "game_answer"
                    ]:
                        st.session_state.pop(key, None)
                    st.rerun()

        else:
            st.subheader(f"Runda {st.session_state.game_round} z 5")
            st.image(st.session_state.game_image, caption="Jaki to rower?", use_container_width=True)

            all_labels = [
                f"{brand.capitalize()} {model.capitalize()}"
                for brand, models in model_classes.items()
                for model in models
            ]
            guess = st.selectbox("Twoja odpowiedź:", all_labels, key=f"guess_{st.session_state.game_round}")

            if not st.session_state.awaiting_next:
                if st.button("✅ Zatwierdź odpowiedź"):
                    if guess == st.session_state.game_answer:
                        st.success("✅ Brawo! Dobra odpowiedź!")
                        st.session_state.game_score += 1
                    else:
                        st.error(f"❌ To było: {st.session_state.game_answer}")

                    st.session_state.awaiting_next = True
                    if st.session_state.game_round < 5:
                        st.session_state.game_round += 1
                        st.session_state.refresh_question = True
                        st.rerun()
                    else:
                        st.session_state.game_finished = True
                        st.rerun()

# Tab 4 - O projekcie
with tab4:
    st.header("ℹ️ O projekcie")

    st.markdown("""
    ### 📅 Praca inżynierska 2025

    **Temat:** Projekt i realizacja systemu rozpoznawania modeli rowerów szosowych na zdjęciach  
    **Autor:** Mikołaj Górny

    ---

    ### 📊 Cel projektu
    Celem pracy było stworzenie systemu, który automatycznie rozpoznaje **model roweru szosowego** na podstawie przesłanego zdjęcia.

    Projekt skupia się na **trzech czołowych markach**:
    - **Trek** (Domane, Madone, Emonda)
    - **Canyon** (Aeroad, Ultimate, Endurance)
    - **Specialized** (Tarmac, Roubaix, Aethos)

    Dla każdej marki przygotowano osobny model klasyfikujący.

    ---

    ### 📊 Techniczne założenia
    - Zdjęcia rowerów pochodzą z internetu i zostały zebrane **ręcznie** do celów naukowych.
    - Wymagana jakość zdjęcia: **ujęcie z boku lub podobny kąt**, **jednolite tło**, dobra widoczność geometrii.
    - Dane zostały przekształcone i uzupełnione przez **augmentacje offline** (odbicia, przesunięcia).
    - Modele zostały wyuczone w oparciu o architekturę **ResNet50**, z wykorzystaniem transfer learningu (fine-tuning).

    ---

    ### 🔄 Użytkowanie
    Aplikacja działa w przeglądarce i pozwala:
    - Rozpoznać model roweru na podstawie przesłanego zdjęcia
    - Automatycznie wykryć, czy zdjęcie zawiera rower (YOLOv8)
    - Przeglądać dostępne modele rowerów z podziałem na marki
    - Zagrać w quiz rozpoznawania rowerów
    - Dodawać posty i komentarze na forum użytkowników

    ---

    ### 🚀 Wykorzystane technologie
    - **Python** – główny język aplikacji
    - **PyTorch, torchvision** – trenowanie i uruchamianie modeli ResNet50
    - **Ultralytics YOLOv8** – detekcja roweru na zdjęciu
    - **Streamlit** – frontend aplikacji i interfejs użytkownika
    - **PIL** – ładowanie i przetwarzanie obrazów
    - **Torchvision transforms** – przekształcanie obrazów wejściowych
    - **os, random** – obsługa plików i danych pomocniczych
    - **Session State** – zarządzanie stanem aplikacji (forum, quiz)


    ---

    ### ⚠️ Uwaga
    Projekt powstał **wyłącznie w celach edukacyjnych**. Zdjęcia rowerów są wykorzystywane zgodnie z prawem cytatu do testów i demonstracji modeli.
    """)

# ======================
# Zakładka: Forum
# ======================
if "forum_posts" not in st.session_state:
    st.session_state.forum_posts = []

with tab5:
    st.header("📢 Forum użytkowników")

    st.subheader("📝 Dodaj nowy post")

    selected_brand = st.selectbox("🚴 Marka roweru:", ["Trek", "Canyon", "Specialized", "Cervelo"], key="brand_post")
    selected_model = st.selectbox(
        "📌 Model roweru:",
        model_classes[selected_brand.lower()],
        key="model_post"
    )

    with st.form("add_post_form", clear_on_submit=True):
        nick = st.text_input("👤 Twój nick:", key="nick_post")
        post_text = st.text_area("✉️ Treść wpisu:", key="post_text")
        submitted = st.form_submit_button("➕ Dodaj post")

        if submitted:
            if nick and post_text:
                new_post = {
                    "nick": nick,
                    "brand": selected_brand,
                    "model": selected_model,
                    "text": post_text,
                    "comments": []
                }
                st.session_state.forum_posts.insert(0, new_post)
                st.success("✅ Post dodany!")
                st.rerun()
            else:
                st.warning("⚠️ Wpisz nick i treść posta.")

    st.markdown("---")
    st.subheader("📬 Posty")

    if not st.session_state.forum_posts:
        st.info("Brak postów. Dodaj swój pierwszy wpis powyżej!")
    else:
        for i, post in enumerate(st.session_state.forum_posts):
            with st.container():
                st.markdown(f"**{post['nick']}** o **{post['brand']} {post['model']}**:")
                st.markdown(f"> {post['text']}")

                if post["comments"]:
                    st.markdown("##### 💬 Komentarze:")
                    for c in post["comments"]:
                        st.markdown(f"- **{c['nick']}**: {c['text']}")

                with st.expander("💬 Dodaj komentarz"):
                    with st.form(f"comment_form_{i}", clear_on_submit=True):
                        comment_nick = st.text_input("👤 Twój nick:", key=f"comment_nick_{i}")
                        comment_text = st.text_input("💬 Twój komentarz:", key=f"comment_text_{i}")
                        submitted_comment = st.form_submit_button("💬 Dodaj komentarz")

                        if submitted_comment:
                            if comment_nick and comment_text:
                                st.session_state.forum_posts[i]["comments"].append({
                                    "nick": comment_nick,
                                    "text": comment_text
                                })
                                st.success("✅ Komentarz dodany!")
                                st.rerun()
                            else:
                                st.warning("⚠️ Wpisz nick i treść komentarza.")

                st.markdown("---")

# ======================
# KONIEC
# ======================
