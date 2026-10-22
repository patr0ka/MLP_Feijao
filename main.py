"""Ponto de entrada: python main.py"""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from carrega_dataset import carrega_dataset
from avaliacao import avalia_modelo, salva_graficos, salva_metricas
from mlp import PerceptronMulticamadas
from preprocessamento import prepara_dados
from treinamento import treina_modelo


PROJECT_DIR = Path(__file__).resolve().parent


def executa_projeto():
    config.valida_config()
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_DIR / dataset_path
    output_root = Path(config.OUTPUT_DIR)
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root

    effective_config = config.para_dicionario()
    effective_config["DATASET_PATH"] = str(dataset_path)
    effective_config["OUTPUT_DIR"] = str(output_root)

    X, y, feature_names = carrega_dataset(
        dataset_path, target_column=config.TARGET_COLUMN,
        expected_features=config.NEURONIOS_ENTRADA, expected_classes=config.NEURONIOS_SAIDA,
    )
    if X.shape[1] != config.NEURONIOS_ENTRADA:
        raise ValueError(
            f"Dataset possui {X.shape[1]} entradas, enquanto NEURONIOS_ENTRADA={config.NEURONIOS_ENTRADA}."
        )

    data = prepara_dados(
        X, y, train_size=config.TRAIN_SIZE, validation_size=config.VALIDATION_SIZE,
        test_size=config.TEST_SIZE, stratified=config.STRATIFIED_SPLIT,
        normalize=config.NORMALIZE_DATA, seed=config.SEED,
    )
    if len(data["class_names"]) != config.NEURONIOS_SAIDA:
        raise ValueError(
            f"Dataset possui {len(data['class_names'])} classes; NEURONIOS_SAIDA={config.NEURONIOS_SAIDA}."
        )
    model = PerceptronMulticamadas(
        input_size=config.NEURONIOS_ENTRADA, hidden_layers=config.CAMADA_INTERMEDIARIA,
        output_size=config.NEURONIOS_SAIDA, hidden_activation=config.HIDDEN_ACTIVATION,
        output_activation=config.OUTPUT_ACTIVATION, seed=config.SEED,
        optimizer=config.OPTIMIZER, learning_rate=config.TAXA_APRENDIZADO,
        l2_lambda=config.L2_LAMBDA,
    )
    print(
        f"\nDataset carregado: {len(X)} amostras, {len(feature_names)} características, "
        f"{len(data['class_names'])} classes.\nIniciando treinamento..."
    )
    training_result = treina_modelo(
        model, data["X_train"], data["y_train"], data["X_validation"], data["y_validation"],
        epochs=config.EPOCAS, batch_size=config.BATCH_SIZE, seed=config.SEED,
        early_stopping=config.EARLY_STOPPING, patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
    )

    # O conjunto de teste é utilizado somente depois de selecionada a melhor época.
    metrics = avalia_modelo(model, data["X_test"], data["y_test"], data["class_names"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for subdirectory in ("models", "metrics", "plots", "logs"):
        (output_root / subdirectory).mkdir(parents=True, exist_ok=True)

    run_config = {
        **effective_config,
        "feature_names": feature_names,
        "class_names": data["class_names"],
        "class_to_index": data["class_to_index"],
        "run_id": run_id,
        "epochs_run": training_result["epochs_run"],
        "best_epoch": training_result["best_epoch"],
    }
    config_snapshot = output_root / "logs" / f"config_{run_id}.json"
    with config_snapshot.open("w", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    if config.SAVE_MODEL:
        scaler = data["scaler"]
        metadata = {
            "class_names": data["class_names"],
            "class_to_index": data["class_to_index"],
            "feature_names": feature_names,
            "normalize_data": config.NORMALIZE_DATA,
            "scaler": None if scaler is None else {
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            },
            "configuration": run_config,
        }
        model_path = output_root / "models" / f"mlp_{run_id}"
        model.salva(model_path, metadata=metadata)

    if config.SAVE_METRICS:
        metrics_path = output_root / "metrics" / run_id
        salva_metricas(metrics, training_result["history"], metrics_path)
    if config.SAVE_PLOTS:
        plots_path = output_root / "plots" / run_id
        salva_graficos(training_result["history"], metrics["confusion_matrix"], data["class_names"], plots_path)

    print(f"\nExecução concluída em {training_result['epochs_run']} época(s). Melhor época: {training_result['best_epoch']}.")
    print(f"Acurácia no teste: {metrics['accuracy']:.4f}")
    print("Relatório de classificação no teste:")
    print(metrics["classification_report_text"])


if __name__ == "__main__":
    try:
        executa_projeto()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}") from None
