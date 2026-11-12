
import os
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "NSL_Dataset", "Train.txt")
TEST_PATH = os.path.join(BASE_DIR, "NSL_Dataset", "Test.txt")
MODEL_PATH = os.path.join(BASE_DIR, "model_clean.pkl")


COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "last_flag"
]

# Map NSL-KDD attack names into five target classes.
ATTACK_MAP = {
    "normal": 0,

    "back": 1, "land": 1, "neptune": 1,
    "pod": 1, "smurf": 1, "teardrop": 1,
    "mailbomb": 1, "apache2": 1, "processtable": 1,
    "udpstorm": 1, "worm": 1,

    "ipsweep": 2, "nmap": 2, "portsweep": 2,
    "satan": 2, "mscan": 2, "saint": 2,

    "ftp_write": 3, "guess_passwd": 3, "imap": 3,
    "multihop": 3, "phf": 3, "spy": 3,
    "warezclient": 3, "warezmaster": 3,
    "snmpgetattack": 3, "named": 3, "xlock": 3,
    "xsnoop": 3, "sendmail": 3, "httptunnel": 3,

    "buffer_overflow": 4, "loadmodule": 4,
    "perl": 4, "rootkit": 4, "ps": 4,
    "sqlattack": 4, "xterm": 4,
"snmpguess": 3,
}

CLASS_NAMES = ["Normal", "DOS", "PROBE", "R2L", "U2R"]

# These features must match the Flask form exactly.
FEATURES = [
    "count",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate",
    "dst_host_srv_count",
    "flag_S0",
    "flag_SF",
    "last_flag",
    "logged_in",
    "same_srv_rate",
    "serror_rate",
    "service_http",
]


def load_dataset(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)

    df["attack"] = df["attack"].astype(str).str.strip().str.lower()
    df["target"] = df["attack"].map(ATTACK_MAP)

    unknown = sorted(df.loc[df["target"].isna(), "attack"].unique())
    if unknown:
        print(f"\nUnmapped attack labels in {os.path.basename(path)}:")
        print(unknown)

    # Do not silently assign unknown attacks to an incorrect class.
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # Derive inputs from actual traffic fields, never from attack labels.
    df["flag_S0"] = (df["flag"] == "S0").astype(int)
    df["flag_SF"] = (df["flag"] == "SF").astype(int)
    df["service_http"] = (df["service"] == "http").astype(int)

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="raise")

    return df


def main():
    print("Loading NSL-KDD datasets...")

    train = load_dataset(TRAIN_PATH)
    test = load_dataset(TEST_PATH)

    print("\nTraining class counts:")
    print(train["target"].map(dict(enumerate(CLASS_NAMES))).value_counts())

    print("\nTest class counts:")
    print(test["target"].map(dict(enumerate(CLASS_NAMES))).value_counts())

    X_train = train[FEATURES]
    y_train = train["target"]

    X_test = test[FEATURES]
    y_test = test["target"]

    # Preserve all five classes despite the dataset's class imbalance.
    model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )
)

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n========== TEST RESULTS ==========")
    print(f"Evaluated test rows: {len(y_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Macro precision: {precision_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Macro recall:    {recall_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Macro F1-score:  {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4]))

    joblib.dump(model, MODEL_PATH)
    print(f"\nNew model saved to: {MODEL_PATH}")
    print("Input feature order:")
    print(FEATURES)
    print("Model classes:", model.classes_)


if __name__ == "__main__":
    main()
