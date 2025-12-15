import cv2
import mediapipe as mp
import csv
import os

# --- KONFIGURACJA ŚCIEŻEK ---
INPUT_FOLDER_PATH = "anon/anon/"  # Folder z plikami wideo
OUTPUT_FOLDER_PATH = "mediapipe_csv/" # Folder na wyniki

# Upewnij się, że folder wyjściowy istnieje
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

DETECTION_CONFIDENCE = 0.5

# Inicjalizacja MediaPipe (robimy to raz przed pętlą dla wydajności)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=DETECTION_CONFIDENCE,
)

LANDMARK_MAPPING = {
    0: "NOSE",
    2: "LEFT_EYE",
    5: "RIGHT_EYE",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE"
}

LANDMARK_NAMES_LIST = list(LANDMARK_MAPPING.values())

# Lista rozszerzeń wideo do przetworzenia
VALID_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

try:
    # Pobierz listę plików w folderze
    files = [f for f in os.listdir(INPUT_FOLDER_PATH) if f.lower().endswith(VALID_EXTENSIONS)]
    total_files = len(files)
    
    print(f"Znaleziono {total_files} plików wideo do przetworzenia.")

    for i, filename in enumerate(files):
        video_path = os.path.join(INPUT_FOLDER_PATH, filename)
        
        # Tworzenie nazwy pliku wyjściowego (zamiana rozszerzenia na .csv)
        base_name = os.path.splitext(filename)[0]
        csv_path = os.path.join(OUTPUT_FOLDER_PATH, f"{base_name}.csv")
        
        print(f"[{i+1}/{total_files}] Przetwarzanie: {filename} -> {os.path.basename(csv_path)}")

        if(os.path.exists(csv_path)):
            print("  Plik CSV już istnieje. Pomijam przetwarzanie.")
            continue

        try:
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['frame_number', "landmark", 'x', 'y', 'z'])

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"  BŁĄD: Nie można otworzyć pliku {filename}. Pomijam.")
                    continue

                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Konwersja koloru
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detekcja pozy
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        for idx, landmark_name in LANDMARK_MAPPING.items():
                            lm = results.pose_landmarks.landmark[idx]
                            # Zapisujemy tylko jeśli widoczność jest wystarczająca
                            if lm.visibility > 0.3:
                                x, y, z = lm.x, lm.y, lm.z
                            else:
                                x, y, z = 0, 0, 0

                            writer.writerow([frame_count, landmark_name, x, y, z])
                    else:
                        # Jeśli na klatce nie wykryto nikogo, wpisz zera dla wszystkich punktów
                        for landmark_name in LANDMARK_NAMES_LIST:
                            writer.writerow([frame_count, landmark_name, 0, 0, 0])

                    frame_count += 1
                    if frame_count % 500 == 0:
                        print(f"  ... przetworzono {frame_count} klatek")

                cap.release()
                print(f"  Zakończono plik: {filename} ({frame_count} klatek)")

        except Exception as e_file:
            print(f"  Wystąpił błąd podczas przetwarzania pliku {filename}: {e_file}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            continue

    print("\n--- Przetwarzanie całego folderu zakończone ---")

except Exception as e:
    print(f"Wystąpił krytyczny błąd skryptu: {e}")

finally:
    pose.close()