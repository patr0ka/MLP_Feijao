from pathlib import Path


# ==================================================
# CONFIGURAÇÕES GERAIS
# ==================================================

SEED = 42
CAMADA_INTERMEDIARIA = [8]
NEURONIOS_ENTRADA = 16
NEURONIOS_SAIDA = 7
TAXA_APRENDIZADO = 0.05

DATASET_PATH = "DryBeanDataset/Dry_Bean_Dataset.xlsx"
TARGET_COLUMN = "Class"

# ==================================================
# DIVISÃO DOS DADOS
# ==================================================

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

STRATIFIED_SPLIT = True
NORMALIZE_DATA = True

# ==================================================
# ARQUITETURA DA MLP
# ==================================================



HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "softmax"

# ==================================================
# TREINAMENTO
# ==================================================

EPOCAS = 100
BATCH_SIZE = 32


OPTIMIZER = "adam"
LOSS_FUNCTION = "cross_entropy"

# ==================================================
# REGULARIZAÇÃO E PARADA
# ==================================================

L2_LAMBDA = 0.0
EARLY_STOPPING = True
PATIENCE = 15
MIN_DELTA = 0.0001

# ==================================================
# SAÍDAS
# ==================================================

SAVE_MODEL = True
SAVE_PLOTS = True
SAVE_METRICS = True

OUTPUT_DIR = "results"


def valida_config() -> None:
    """Valida configurações independentes do conteúdo do dataset."""
    ratios = (TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE)
    if any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError("TRAIN_SIZE, VALIDATION_SIZE e TEST_SIZE devem estar entre 0 e 1.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE deve ser igual a 1.")
    if NEURONIOS_ENTRADA <= 0 or NEURONIOS_SAIDA <= 1:
        raise ValueError("NEURONIOS_ENTRADA deve ser positivo e NEURONIOS_SAIDA deve ser maior que 1.")
    if not CAMADA_INTERMEDIARIA or any(not isinstance(size, int) or size <= 0 for size in CAMADA_INTERMEDIARIA):
        raise ValueError("CAMADA_INTERMEDIARIA deve conter pelo menos um número inteiro positivo.")
    if HIDDEN_ACTIVATION.lower() not in {"relu", "sigmoid"}:
        raise ValueError("HIDDEN_ACTIVATION deve ser 'relu' ou 'sigmoid'.")
    if OUTPUT_ACTIVATION.lower() != "softmax":
        raise ValueError("Para classificação multiclasse, OUTPUT_ACTIVATION deve ser 'softmax'.")
    if OPTIMIZER.lower() not in {"adam", "sgd"}:
        raise ValueError("OPTIMIZER não suportado. Utilize 'adam' ou 'sgd'.")
    if LOSS_FUNCTION.lower() != "cross_entropy":
        raise ValueError("LOSS_FUNCTION não suportada. Utilize 'cross_entropy'.")
    if EPOCAS <= 0 or BATCH_SIZE <= 0 or TAXA_APRENDIZADO <= 0:
        raise ValueError("EPOCAS, BATCH_SIZE e TAXA_APRENDIZADO devem ser positivos.")
    if L2_LAMBDA < 0 or PATIENCE <= 0 or MIN_DELTA < 0:
        raise ValueError("L2_LAMBDA e MIN_DELTA não podem ser negativos; PATIENCE deve ser positivo.")


def para_dicionario() -> dict:
    """Retorna uma cópia serializável de todas as configurações públicas."""
    return {
        name: value
        for name, value in globals().copy().items()
        if name.isupper() and not isinstance(value, (Path,))
    }
