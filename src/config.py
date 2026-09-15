from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "pilotia_activity.csv"

RANDOM_SEED = 42
DEFAULT_ROWS = 120_000

SERVICES = [
    "Indemnités journalières",
    "Remboursements",
    "Affiliation",
    "Complémentaire santé",
    "Accidents du travail",
]

CAISSES = [f"Caisse {i:02d}" for i in range(1, 9)]

REQUEST_TYPES = [
    "Demande standard",
    "Réclamation",
    "Mise à jour dossier",
    "Ouverture de droits",
    "Justificatif",
]
