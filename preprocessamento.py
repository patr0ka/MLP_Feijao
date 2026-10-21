"""Divisão estratificada, codificação de classes e normalização dos dados."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepara_dados(X, y, train_size=0.70, validation_size=0.15, test_size=0.15,
                 stratified=True, normalize=True, seed=42):
    """Divide os dados em treino/validação/teste e ajusta scaler somente no treino."""
    if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
        raise ValueError("As proporções de treino, validação e teste devem somar 1.")
    class_names = sorted(str(label) for label in set(y))
    if len(class_names) < 2:
        raise ValueError("São necessárias pelo menos duas classes para treinar a MLP.")
    class_to_index = {label: index for index, label in enumerate(class_names)}
    y_encoded = np.asarray([class_to_index[str(label)] for label in y], dtype=np.int64)
    X_values = np.asarray(X, dtype=np.float64)

    stratify_all = y_encoded if stratified else None
    try:
        X_train, X_other, y_train, y_other = train_test_split(
            X_values, y_encoded, train_size=train_size, random_state=seed,
            shuffle=True, stratify=stratify_all,
        )
        validation_fraction_of_other = validation_size / (validation_size + test_size)
        stratify_other = y_other if stratified else None
        X_validation, X_test, y_validation, y_test = train_test_split(
            X_other, y_other, train_size=validation_fraction_of_other,
            random_state=seed, shuffle=True, stratify=stratify_other,
        )
    except ValueError as exc:
        raise ValueError(
            "Não foi possível dividir os dados com as proporções/estratificação configuradas. "
            "Verifique a quantidade de exemplos de cada classe e os tamanhos dos conjuntos."
        ) from exc

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)

    return {
        "X_train": np.asarray(X_train, dtype=np.float64),
        "X_validation": np.asarray(X_validation, dtype=np.float64),
        "X_test": np.asarray(X_test, dtype=np.float64),
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "scaler": scaler,
    }
