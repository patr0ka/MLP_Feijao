"""Carregamento e validação do Dry Bean Dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


def carrega_dataset(path: str | Path, target_column: str = "Class", expected_features: int = 16,
                 expected_classes: int = 7) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Carrega o arquivo, valida as colunas e retorna X, y e os nomes das entradas."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {dataset_path}. Coloque o arquivo nesse caminho "
            "ou ajuste DATASET_PATH em config.py."
        )

    suffix = dataset_path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(dataset_path)
        elif suffix == ".csv":
            frame = pd.read_csv(dataset_path)
        else:
            raise ValueError(f"Formato '{suffix}' não suportado. Utilize .xlsx, .xls ou .csv.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Não foi possível ler o dataset '{dataset_path}': {exc}") from exc

    if frame.empty:
        raise ValueError("O arquivo do dataset não contém registros.")

    normalized_names = {str(column).strip().casefold(): column for column in frame.columns}
    target_key = str(target_column).strip().casefold()
    if target_key not in normalized_names:
        raise ValueError(
            f"Coluna-alvo '{target_column}' não encontrada. Colunas disponíveis: "
            f"{list(frame.columns)}. Ajuste TARGET_COLUMN em config.py."
        )
    actual_target = normalized_names[target_key]

    feature_columns = []
    missing_features = []
    for expected in EXPECTED_FEATURES:
        actual = normalized_names.get(expected.casefold())
        if actual is None or actual == actual_target:
            missing_features.append(expected)
        else:
            feature_columns.append(actual)
    if missing_features:
        raise ValueError(
            "O dataset não contém todas as 16 características esperadas. "
            f"Colunas ausentes: {missing_features}. Colunas disponíveis: {list(frame.columns)}"
        )
    if len(feature_columns) != expected_features:
        raise ValueError(
            f"Foram identificadas {len(feature_columns)} características, mas NEURONIOS_ENTRADA="
            f"{expected_features}. Ajuste config.py ou verifique o arquivo."
        )

    if frame[actual_target].isna().any():
        count = int(frame[actual_target].isna().sum())
        raise ValueError(f"A coluna-alvo contém {count} valor(es) ausente(s). Corrija o arquivo antes do treino.")

    X = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        try:
            X[column] = pd.to_numeric(X[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"A característica '{column}' contém valores não numéricos.") from exc
    if X.isna().any().any():
        missing = {str(key): int(value) for key, value in X.isna().sum().items() if value}
        raise ValueError(f"O dataset contém valores ausentes nas características: {missing}")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("O dataset contém valores infinitos ou inválidos nas características.")

    y = frame[actual_target].astype(str).str.strip()
    if (y == "").any():
        raise ValueError("A coluna-alvo contém rótulos vazios.")
    classes = sorted(y.unique().tolist())
    if len(classes) != expected_classes:
        raise ValueError(
            f"Esperadas {expected_classes} classes distintas, mas foram encontradas {len(classes)}: {classes}."
        )
    return X, y, [str(column) for column in feature_columns]
