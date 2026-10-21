import numpy as np

from perceptron_multicamadas import PerceptronMulticamadas
from treinamento import treina_modelo


def test_propagacao_perda_gradientes_e_atualizacao():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 16))
    y = np.arange(12) % 7
    model = PerceptronMulticamadas(16, [16, 8], 7, seed=19)

    probabilities, cache = model.propaga_para_frente(X, return_cache=True)
    assert probabilities.shape == (12, 7)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(12), atol=1e-12)
    loss = model.perda(y, probabilities)
    assert np.isfinite(loss)

    gradients = model.retropropaga(y, cache)
    assert all(np.isfinite(value).all() for value in gradients["weights"])
    assert all(np.isfinite(value).all() for value in gradients["biases"])
    before = [weight.copy() for weight in model.weights]
    model.atualiza_parametros(gradients)
    assert any(not np.array_equal(old, new) for old, new in zip(before, model.weights))
    assert [weight.shape for weight in model.weights] == [(16, 16), (16, 8), (8, 7)]


def test_conjunto_artificial_aprende_e_modelo_pode_ser_recarregado(tmp_path):
    rng = np.random.default_rng(3)
    centers = np.array([[-2.0, -2.0], [2.0, -2.0], [0.0, 2.0]])
    X = np.vstack([center + rng.normal(scale=0.45, size=(45, 2)) for center in centers])
    y = np.repeat(np.arange(3), 45)
    model = PerceptronMulticamadas(2, [12], 3, seed=5, learning_rate=0.025)
    initial_loss = model.perda(y, model.preve_probabilidades(X))
    result = treina_modelo(
        model, X, y, X, y, epochs=60, batch_size=24, seed=5,
        early_stopping=False, verbose=False,
    )
    final_loss = model.perda(y, model.preve_probabilidades(X))
    assert result["epochs_run"] == 60
    assert final_loss < initial_loss

    metadata = {
        "class_names": ["Barbunya", "Seker", "Sira"],
        "scaler": {"mean": [0.0, 0.0], "scale": [1.0, 1.0]},
    }
    model.salva(tmp_path / "saved", metadata)
    restored, restored_metadata = PerceptronMulticamadas.carrega(tmp_path / "saved")
    np.testing.assert_allclose(restored.preve_probabilidades(X), model.preve_probabilidades(X))
    _, labels = restored.preve_dados_brutos(X[:4], restored_metadata)
    assert all(label in metadata["class_names"] for label in labels)


def test_inicializacao_e_reprodutivel():
    first = PerceptronMulticamadas(4, [5], 3, seed=123)
    second = PerceptronMulticamadas(4, [5], 3, seed=123)
    for left, right in zip(first.weights, second.weights):
        np.testing.assert_array_equal(left, right)
