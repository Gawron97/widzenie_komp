import glob

# Ścieżki
CSV_GLOB_PATTERN = "data/data/mp_pose_csv/*.csv"
LABELS_DIR = "data/data/label"

# Parametry danych
SEQ_LEN = 32
STEP = 8
NUM_FEATURES = 2  # (x, y)
BATCH_SIZE = 64  # Increased for better GPU utilization

# Definicja szkieletu (podzbiór MediaPipe)
JOINT_NAMES = [
    "NOSE",            # 0
    "LEFT_SHOULDER",   # 1
    "RIGHT_SHOULDER",  # 2
    "LEFT_ELBOW",      # 3
    "RIGHT_ELBOW",     # 4
    "LEFT_WRIST",      # 5
    "RIGHT_WRIST",     # 6
    "LEFT_HIP",        # 7
    "RIGHT_HIP",       # 8
    "LEFT_KNEE",       # 9
    "RIGHT_KNEE",      # 10
    "LEFT_ANKLE",      # 11
    "RIGHT_ANKLE",     # 12
]

HIP_INDICES = [7, 8]  # Indeksy stawów bioder

# Pary połączeń (krawędzie grafu)
SKELETON_PAIRS = [
    ("NOSE", "LEFT_SHOULDER"), ("NOSE", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
]

def get_csv_paths():
    paths = sorted(glob.glob(CSV_GLOB_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No CSV files found matching: {CSV_GLOB_PATTERN}")
    return paths