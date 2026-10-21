import numpy as np
import pandas as pd
import pytest

from carrega_dataset import EXPECTED_FEATURES, carrega_dataset
from preprocessamento import prepara_dados


def cria_quadro(rows_per_class=12):
    rng = np.random.default_rng(10)
    class_names = ["Seker", "Barbunya", "Bombay", "Cali", "Dermason", "Horoz", "Sira"]
    frame = pd.DataFrame(rng.normal(size=(rows_per_class * len(class_names), len(EXPECTED_FEATURES))),
                         columns=EXPECTED_FEATURES)
    frame["Class"] = np.repeat(class_names, rows_per_class)
    # Embaralha a ordem das colunas para garantir que o carregador seleciona por nome.
    return frame[["Class", *reversed(EXPECTED_FEATURES)]]


def test_dataset_e_carregado_pelos_nomes_das_colunas(tmp_path):
    path = tmp_path / "beans.csv"
    cria_quadro().to_csv(path, index=False)
    X, y, feature_names = carrega_dataset(path)
    assert X.shape == (84, 16)
    assert y.nunique() == 7
    assert "Class" not in feature_names
    assert set(feature_names) == set(EXPECTED_FEATURES)


def test_carregador_identifica_arquivo_ausente_e_valores_ausentes(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset não encontrado"):
        carrega_dataset(tmp_path / "missing.xlsx")

    frame = cria_quadro()
    frame.loc[0, EXPECTED_FEATURES[0]] = np.nan
    path = tmp_path / "missing_value.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="valores ausentes"):
        carrega_dataset(path)


def test_divisao_e_normalizacao_sao_reprodutiveis():
    frame = cria_quadro(rows_per_class=20)
    X, y, _ = extrai_dados_para_divisao(frame)
    first = prepara_dados(X, y, seed=44, normalize=True)
    second = prepara_dados(X, y, seed=44, normalize=True)
    np.testing.assert_array_equal(first["X_train"], second["X_train"])
    np.testing.assert_array_equal(first["y_test"], second["y_test"])
    np.testing.assert_allclose(first["X_train"].mean(axis=0), np.zeros(16), atol=1e-12)


def extrai_dados_para_divisao(frame):
    """Mantém o teste da divisão independente da criação de um segundo arquivo XLSX."""
    X = frame.loc[:, EXPECTED_FEATURES].copy()
    y = frame["Class"].copy()
    return X, y, EXPECTED_FEATURES
