"""Métricas e gráficos de avaliação usando somente o conjunto de teste."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def avalia_modelo(model, X_test, y_test, class_names):
    """Calcula métricas finais; y_test contém índices na ordem de class_names."""
    probabilities = model.preve_probabilidades(X_test)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.arange(len(class_names))
    report = classification_report(
        y_test, predictions, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_test, predictions, labels=labels, target_names=class_names, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
        "classification_report_text": report_text,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def salva_metricas(metrics, history, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    serializable_metrics = {
        key: value for key, value in metrics.items()
        if key not in {"predictions", "probabilities"}
    }
    with (directory / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(serializable_metrics, file, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    report = metrics["classification_report"]
    pd.DataFrame(report).transpose().to_csv(directory / "classification_report.csv", index_label="class")
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(directory / "confusion_matrix.csv", index=False)


def salva_graficos(history, confusion, class_names, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Treino")
    ax.plot(epochs, history["validation_loss"], label="Validação")
    ax.set(title="Perda por época", xlabel="Época", ylabel="Entropia cruzada")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(directory / "loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_accuracy"], label="Treino")
    ax.plot(epochs, history["validation_accuracy"], label="Validação")
    ax.set(title="Acurácia por época", xlabel="Época", ylabel="Acurácia")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(directory / "accuracy.png", dpi=150)
    plt.close(fig)

    matrix = np.asarray(confusion, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Quantidade")
    positions = np.arange(len(class_names))
    ax.set(
        title="Matriz de confusão — conjunto de teste",
        xlabel="Classe prevista",
        ylabel="Classe real",
        xticks=positions,
        yticks=positions,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(fig)
