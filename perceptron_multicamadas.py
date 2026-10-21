"""Rede Perceptron Multicamadas multiclasse implementada manualmente com NumPy."""

import json
from pathlib import Path

import numpy as np


class PerceptronMulticamadas:
    """Perceptron multicamadas com ReLU/sigmoide, softmax, entropia cruzada, SGD ou Adam."""

    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int,
                 hidden_activation: str = "relu", output_activation: str = "softmax",
                 seed: int = 42, optimizer: str = "adam", learning_rate: float = 0.001,
                 l2_lambda: float = 0.0):
        self.input_size = int(input_size)
        self.hidden_layers = [int(size) for size in hidden_layers]
        self.output_size = int(output_size)
        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower()
        self.seed = int(seed)
        self.optimizer = optimizer.lower()
        self.learning_rate = float(learning_rate)
        self.l2_lambda = float(l2_lambda)
        if self.input_size <= 0 or self.output_size <= 1 or any(size <= 0 for size in self.hidden_layers):
            raise ValueError("Os tamanhos das camadas devem ser positivos e a saída deve ter ao menos 2 neurônios.")
        if self.hidden_activation not in {"relu", "sigmoid"}:
            raise ValueError("Ativação oculta não suportada. Utilize 'relu' ou 'sigmoid'.")
        if self.output_activation != "softmax":
            raise ValueError("A ativação de saída suportada para classificação é 'softmax'.")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("Otimizador não suportado. Utilize 'adam' ou 'sgd'.")
        if self.learning_rate <= 0 or self.l2_lambda < 0:
            raise ValueError("learning_rate deve ser positivo e l2_lambda não pode ser negativo.")

        self.layer_sizes = [self.input_size, *self.hidden_layers, self.output_size]
        rng = np.random.default_rng(self.seed)
        self.weights = []
        self.biases = []
        for layer_index, (fan_in, fan_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            is_hidden = layer_index < len(self.layer_sizes) - 2
            activation = self.hidden_activation if is_hidden else "softmax"
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)  # Inicialização He
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier para sigmoid/softmax
            self.weights.append(rng.standard_normal((fan_in, fan_out)) * scale)
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

        self._adam_step = 0
        self._adam_m_w = [np.zeros_like(weight) for weight in self.weights]
        self._adam_v_w = [np.zeros_like(weight) for weight in self.weights]
        self._adam_m_b = [np.zeros_like(bias) for bias in self.biases]
        self._adam_v_b = [np.zeros_like(bias) for bias in self.biases]

    @staticmethod
    def _ativa_relu(values):
        return np.maximum(0.0, values)

    @staticmethod
    def _ativa_sigmoide(values):
        clipped = np.clip(values, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _ativa_softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _ativa_oculta(self, values):
        if self.hidden_activation == "relu":
            return self._ativa_relu(values)
        return self._ativa_sigmoide(values)

    def _derivada_ativacao_oculta(self, pre_activation, activation):
        if self.hidden_activation == "relu":
            return (pre_activation > 0.0).astype(np.float64)
        return activation * (1.0 - activation)

    def propaga_para_frente(self, X, return_cache: bool = False):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(f"Esperada uma matriz com {self.input_size} características por amostra; recebido {X.shape}.")
        if not np.isfinite(X).all():
            raise ValueError("As entradas da rede devem ser valores finitos.")
        activations = [X]
        pre_activations = []
        current = X
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            pre_activations.append(z)
            if index == len(self.weights) - 1:
                current = self._ativa_softmax(z)
            else:
                current = self._ativa_oculta(z)
            activations.append(current)
        if return_cache:
            return current, {"activations": activations, "pre_activations": pre_activations}
        return current

    def perda(self, y_true, probabilities, include_regularization: bool = True) -> float:
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape != (len(y_true), self.output_size):
            raise ValueError("As dimensões dos rótulos e das probabilidades não correspondem.")
        if np.any(y_true < 0) or np.any(y_true >= self.output_size):
            raise ValueError("Índice de classe fora do intervalo de saídas da rede.")
        epsilon = np.finfo(np.float64).eps
        loss = -np.mean(np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], epsilon, 1.0)))
        if include_regularization and self.l2_lambda > 0:
            loss += 0.5 * self.l2_lambda * sum(float(np.sum(weight ** 2)) for weight in self.weights)
        return float(loss)

    def retropropaga(self, y_true, cache):
        """Calcula gradientes via regra da cadeia; softmax + CE usa o delta simplificado."""
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]
        probabilities = activations[-1]
        if len(y_true) != probabilities.shape[0]:
            raise ValueError("A quantidade de rótulos não corresponde ao batch.")
        if np.any(y_true < 0) or np.any(y_true >= self.output_size):
            raise ValueError("Índice de classe fora do intervalo de saídas da rede.")

        batch_size = len(y_true)
        delta = probabilities.copy()
        delta[np.arange(batch_size), y_true] -= 1.0
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)
        for layer_index in range(len(self.weights) - 1, -1, -1):
            gradients_w[layer_index] = activations[layer_index].T @ delta / batch_size
            gradients_b[layer_index] = np.sum(delta, axis=0, keepdims=True) / batch_size
            if self.l2_lambda > 0:
                gradients_w[layer_index] += self.l2_lambda * self.weights[layer_index]
            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * self._derivada_ativacao_oculta(
                    pre_activations[layer_index - 1], activations[layer_index]
                )
        return {"weights": gradients_w, "biases": gradients_b}

    def atualiza_parametros(self, gradients):
        if self.optimizer == "sgd":
            for index in range(len(self.weights)):
                self.weights[index] -= self.learning_rate * gradients["weights"][index]
                self.biases[index] -= self.learning_rate * gradients["biases"][index]
            return

        self._adam_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for index in range(len(self.weights)):
            grad_w = gradients["weights"][index]
            grad_b = gradients["biases"][index]
            self._adam_m_w[index] = beta1 * self._adam_m_w[index] + (1.0 - beta1) * grad_w
            self._adam_v_w[index] = beta2 * self._adam_v_w[index] + (1.0 - beta2) * (grad_w ** 2)
            self._adam_m_b[index] = beta1 * self._adam_m_b[index] + (1.0 - beta1) * grad_b
            self._adam_v_b[index] = beta2 * self._adam_v_b[index] + (1.0 - beta2) * (grad_b ** 2)
            m_w_hat = self._adam_m_w[index] / (1.0 - beta1 ** self._adam_step)
            v_w_hat = self._adam_v_w[index] / (1.0 - beta2 ** self._adam_step)
            m_b_hat = self._adam_m_b[index] / (1.0 - beta1 ** self._adam_step)
            v_b_hat = self._adam_v_b[index] / (1.0 - beta2 ** self._adam_step)
            self.weights[index] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.biases[index] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def treina_lote(self, X, y):
        probabilities, cache = self.propaga_para_frente(X, return_cache=True)
        loss = self.perda(y, probabilities)
        gradients = self.retropropaga(y, cache)
        self.atualiza_parametros(gradients)
        return loss

    def preve_probabilidades(self, X):
        return self.propaga_para_frente(X)

    def preve_classes(self, X):
        return np.argmax(self.preve_probabilidades(X), axis=1)

    def estado_parametros(self):
        return {
            "weights": [weight.copy() for weight in self.weights],
            "biases": [bias.copy() for bias in self.biases],
        }

    def carrega_estado_parametros(self, state):
        if len(state["weights"]) != len(self.weights) or len(state["biases"]) != len(self.biases):
            raise ValueError("O estado salvo não corresponde à arquitetura desta rede.")
        for index, (weight, bias) in enumerate(zip(state["weights"], state["biases"])):
            if weight.shape != self.weights[index].shape or bias.shape != self.biases[index].shape:
                raise ValueError(f"Dimensão incompatível nos parâmetros da camada {index}.")
        self.weights = [np.asarray(value, dtype=np.float64).copy() for value in state["weights"]]
        self.biases = [np.asarray(value, dtype=np.float64).copy() for value in state["biases"]]

    def salva(self, directory: str | Path, metadata: dict | None = None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f"weight_{index}"] = weight
            arrays[f"bias_{index}"] = bias
        np.savez_compressed(directory / "mlp_parameters.npz", **arrays)
        full_metadata = {
            "architecture": self.layer_sizes,
            "hidden_layers": self.hidden_layers,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "seed": self.seed,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "l2_lambda": self.l2_lambda,
            "parameter_file": "mlp_parameters.npz",
        }
        if metadata:
            full_metadata.update(metadata)
        with (directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(full_metadata, file, ensure_ascii=False, indent=2)
        return directory

    @classmethod
    def carrega(cls, directory: str | Path):
        directory = Path(directory)
        metadata_path = directory if directory.is_file() else directory / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        directory = metadata_path.parent
        architecture = metadata["architecture"]
        model = cls(
            input_size=architecture[0], hidden_layers=architecture[1:-1], output_size=architecture[-1],
            hidden_activation=metadata["hidden_activation"], output_activation=metadata["output_activation"],
            seed=metadata.get("seed", 42), optimizer=metadata.get("optimizer", "adam"),
            learning_rate=metadata.get("learning_rate", 0.001), l2_lambda=metadata.get("l2_lambda", 0.0),
        )
        parameter_path = directory / metadata.get("parameter_file", "mlp_parameters.npz")
        with np.load(parameter_path, allow_pickle=False) as saved:
            state = {
                "weights": [saved[f"weight_{i}"] for i in range(len(model.weights))],
                "biases": [saved[f"bias_{i}"] for i in range(len(model.biases))],
            }
        model.carrega_estado_parametros(state)
        return model, metadata

    def preve_dados_brutos(self, X, metadata: dict):
        """Aplica o scaler salvo, se houver, e retorna probabilidades e nomes das classes."""
        values = np.asarray(X, dtype=np.float64)
        scaler_info = metadata.get("scaler")
        if scaler_info is not None:
            values = (values - np.asarray(scaler_info["mean"], dtype=np.float64)) / np.asarray(
                scaler_info["scale"], dtype=np.float64
            )
        probabilities = self.preve_probabilidades(values)
        classes = metadata.get("class_names", [])
        labels = [classes[index] for index in np.argmax(probabilities, axis=1)] if classes else self.preve_classes(values)
        return probabilities, labels
