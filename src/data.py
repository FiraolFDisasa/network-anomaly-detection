"""Loading and preprocessing of the NSL-KDD dataset.

NSL-KDD files (KDDTrain+.txt, KDDTest+.txt) have 41 features, a label and a
difficulty score, with no header row. The test set contains attack types that
never appear in the training set, which lets us measure detection of unseen
("zero-day") attacks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label", "difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]
LOG_SCALED = ["duration", "src_bytes", "dst_bytes"]  # heavy-tailed
DROPPED = ["num_outbound_cmds"]  # constant zero in NSL-KDD

ATTACK_CATEGORIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
            "udpstorm", "processtable", "mailbomb", "worm"],
    "Probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "R2L": ["guess_passwd", "ftp_write", "imap", "phf", "multihop",
            "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
            "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"],
    "U2R": ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
            "xterm", "ps"],
}
_TYPE_TO_CATEGORY = {t: c for c, types in ATTACK_CATEGORIES.items() for t in types}


def attack_category(label: str) -> str:
    if label == "normal":
        return "Normal"
    return _TYPE_TO_CATEGORY.get(label, "Other")


def load_nsl_kdd(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read KDDTrain+.txt and KDDTest+.txt from data_dir."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "KDDTrain+.txt", header=None, names=COLUMNS)
    test = pd.read_csv(data_dir / "KDDTest+.txt", header=None, names=COLUMNS)
    return _annotate(train), _annotate(test)


def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].str.strip().str.rstrip(".")
    df["category"] = df["label"].map(attack_category)
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def make_synthetic(n_train: int = 20000, n_test: int = 8000, seed: int = 42):
    """Generate NSL-KDD-shaped synthetic data so the pipeline can be tested
    without downloading the real dataset. Results on it mean nothing."""
    rng = np.random.default_rng(seed)
    numeric = [f for f in FEATURES if f not in CATEGORICAL]

    def block(n, label, shift):
        rows = {f: np.abs(rng.normal(shift, 1.0, n)) for f in numeric}
        rows["src_bytes"] = np.exp(rng.normal(6 + shift, 1.5, n))
        rows["dst_bytes"] = np.exp(rng.normal(7 - shift, 1.5, n))
        rows["num_outbound_cmds"] = np.zeros(n)
        rows["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
        rows["service"] = rng.choice(["http", "smtp", "ftp", "private"], n)
        rows["flag"] = rng.choice(["SF", "S0", "REJ"], n,
                                  p=[0.9, 0.05, 0.05] if label == "normal" else None)
        rows["label"] = label
        rows["difficulty"] = 20
        return pd.DataFrame(rows)

    train = pd.concat([
        block(int(n_train * 0.55), "normal", 0.0),
        block(int(n_train * 0.25), "neptune", 2.5),
        block(int(n_train * 0.15), "satan", 1.5),
        block(int(n_train * 0.05), "guess_passwd", 1.0),
    ])
    # "mscan" and "processtable" appear only in the test set (unseen attacks).
    test = pd.concat([
        block(int(n_test * 0.45), "normal", 0.0),
        block(int(n_test * 0.20), "neptune", 2.5),
        block(int(n_test * 0.15), "mscan", 1.2),
        block(int(n_test * 0.10), "processtable", 0.8),
        block(int(n_test * 0.10), "guess_passwd", 1.0),
    ])
    shuffle = lambda d: d.sample(frac=1, random_state=seed).reset_index(drop=True)
    return _annotate(shuffle(train)), _annotate(shuffle(test))


class Preprocessor:
    """One-hot encodes categorical features, log-scales heavy-tailed ones and
    min-max scales everything. Fitted on normal training traffic only, so no
    information about attacks or the test set leaks into preprocessing."""

    def __init__(self):
        numeric = [f for f in FEATURES if f not in CATEGORICAL + DROPPED]
        self.transformer = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", MinMaxScaler(clip=True), numeric),
        ])

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df[FEATURES].copy()
        df[LOG_SCALED] = np.log1p(df[LOG_SCALED].clip(lower=0))
        return df

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.transformer.fit(self._prepare(df))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transformer.transform(self._prepare(df)).astype(np.float32)
