"""Laço de treinamento da rede com lotes, histórico e parada antecipada."""

import numpy as np


def treina_modelo(model, X_train, y_train, X_validation, y_validation, *, epochs=100,
                batch_size=32, seed=42, early_stopping=True, patience=15,
                min_delta=0.0001, verbose=True):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_validation = np.asarray(X_validation, dtype=np.float64)
    y_validation = np.asarray(y_validation, dtype=np.int64)
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("As quantidades de exemplos e rótulos devem corresponder.")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs e batch_size devem ser positivos.")

    rng = np.random.default_rng(seed)
    history = {"epoch": [], "train_loss": [], "validation_loss": [],
               "train_accuracy": [], "validation_accuracy": []}
    best_loss = float("inf")
    best_state = model.estado_parametros()
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            model.treina_lote(X_train[batch_indices], y_train[batch_indices])

        train_probabilities = model.preve_probabilidades(X_train)
        validation_probabilities = model.preve_probabilidades(X_validation)
        train_loss = model.perda(y_train, train_probabilities)
        validation_loss = model.perda(y_validation, validation_probabilities)
        train_accuracy = float(np.mean(np.argmax(train_probabilities, axis=1) == y_train))
        validation_accuracy = float(np.mean(np.argmax(validation_probabilities, axis=1) == y_validation))
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_state = model.estado_parametros()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Época {epoch:03d}/{epochs}: "
                f"perda treino={train_loss:.4f}, validação={validation_loss:.4f}, "
                f"acurácia treino={train_accuracy:.4f}, validação={validation_accuracy:.4f}"
            )
        if early_stopping and epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping: validação sem melhora por {patience} épocas.")
            break

    model.carrega_estado_parametros(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "epochs_run": len(history["epoch"])}
