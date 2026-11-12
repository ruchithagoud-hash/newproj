"""Shared definitions for training (train_compare.py) and inference (app.py).

Keeping the label mapping, feature list, feature encoding and risk levels in ONE
place guarantees that the Flask app feeds the models exactly what they were
trained on (same features, same order, same encoding).
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "NSL_Dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42

# The 43 columns of Train.txt / Test.txt (no header row in the files).
# The 42nd column is the attack label; the 43rd is NSL-KDD's "difficulty level"
# (the notebook called it `last_flag`).
RAW_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag",
]

# ---------------------------------------------------------------- classes ----
CLASS_NAMES = {0: "Normal", 1: "DOS", 2: "PROBE", 3: "R2L", 4: "U2R"}
CLASS_IDS = sorted(CLASS_NAMES)
CLASS_LABELS = [CLASS_NAMES[i] for i in CLASS_IDS]

# Risk levels are application-defined (not part of the dataset).
RISK_LEVELS = {0: "Low Risk", 1: "High Risk", 2: "Medium Risk", 3: "High Risk", 4: "Critical Risk"}

# Standard NSL-KDD attack-family grouping (train + KDDTest+ labels).
ATTACK_FAMILIES = {
    0: ["normal"],
    1: ["back", "land", "neptune", "pod", "smurf", "teardrop",
        "apache2", "udpstorm", "processtable", "worm", "mailbomb"],
    2: ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    3: ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
        "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
        "httptunnel", "sendmail", "named"],
    4: ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"],
}
ATTACK_MAP = {name: cid for cid, names in ATTACK_FAMILIES.items() for name in names}

# --------------------------------------------------------------- features ----
# PRIMARY feature set used for BOTH models (and by the Flask app).
# `last_flag` from the original 12-feature set is deliberately excluded: it is
# NSL-KDD's difficulty score (how many reference classifiers got the record
# right), an annotation that is not observable network traffic.
FEATURES = [
    "count",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate",
    "dst_host_srv_count",
    "flag_S0",
    "flag_SF",
    "logged_in",
    "same_srv_rate",
    "serror_rate",
    "service_http",
]

# The student's original experimental set (kept only for a side experiment).
FEATURES_WITH_DIFFICULTY = [
    "count", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate", "dst_host_srv_count", "flag_S0", "flag_SF",
    "last_flag", "logged_in", "same_srv_rate", "serror_rate", "service_http",
]

# Columns that must never be model inputs (derived from / identical to the target).
FORBIDDEN_FEATURES = {"attack", "attack_class", "y", "last_flag"}


def is_leaky(feature: str) -> bool:
    return feature in FORBIDDEN_FEATURES or feature.startswith("attack_")


def assert_no_leakage(features):
    bad = [f for f in features if is_leaky(f)]
    if bad:
        raise ValueError(f"Target-leaking / non-traffic features present: {bad}")


# ---------------------------------------------------------------- loading ----
def load_raw(split: str) -> pd.DataFrame:
    """Load Train.txt or Test.txt. header=None: the files have no header row
    (the original notebook silently used the first record as the header)."""
    path = DATA_DIR / f"{split}.txt"
    return pd.read_csv(path, header=None, names=RAW_COLUMNS)


def encode_flag(flag: str):
    """flag -> (flag_S0, flag_SF); every other TCP flag is encoded (0, 0)."""
    return int(flag == "S0"), int(flag == "SF")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the indicator columns used by the models (vectorised twin of
    encode_flag / the Flask form encoding)."""
    out = df.copy()
    out["flag_S0"] = (out["flag"] == "S0").astype(int)
    out["flag_SF"] = (out["flag"] == "SF").astype(int)
    out["service_http"] = (out["service"] == "http").astype(int)
    return out


def map_labels(df: pd.DataFrame) -> pd.Series:
    """Map attack names to class ids. Raises (never drops) on unmapped labels."""
    y = df["attack"].map(ATTACK_MAP)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), "attack"].unique())
        raise ValueError(f"Unmapped attack labels: {unknown}")
    return y.astype(int)
