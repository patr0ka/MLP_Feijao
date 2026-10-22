# Classificação de variedades de feijão com uma MLP

Projeto acadêmico em Python que classifica as sete variedades do Dry Bean Dataset usando uma rede Perceptron multicamadas (MLP). A rede, incluindo propagação para frente, cálculo da perda, retropropagação e atualização dos parâmetros, é implementada manualmente com NumPy. Scikit-learn é usado somente para divisão e normalização dos dados e para métricas.

O programa recebe descritores numéricos já extraídos dos grãos. Ele não processa imagens.

## Dados

O arquivo de dados não é armazenado neste repositório. A configuração inicial espera o arquivo:

    DryBeanDataset/Dry_Bean_Dataset.xlsx

Crie a pasta localmente e coloque nela um arquivo autorizado que contenha as dezesseis características numéricas esperadas e a coluna alvo Class. O carregador também aceita arquivos XLS e CSV. As colunas são identificadas pelo nome, sem depender da ordem, e precisam corresponder aos nomes do Dry Bean Dataset.

O Dry Bean Dataset pode ser obtido no [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean). Os arquivos de uma competição privada do Kaggle podem exigir acesso autorizado pela conta do participante. Não publique no GitHub arquivos obtidos sob acesso privado. Se usar o train.csv da competição, confirme que ele contém a coluna Class e todas as dezesseis características esperadas. O programa divide internamente esse arquivo em treino, validação e teste. O test.csv de uma competição normalmente não tem rótulos e o sample_submission.csv é apenas um exemplo de formato de envio; eles não substituem o arquivo rotulado usado para treinar e avaliar este projeto.

Para usar outro caminho ou nome de coluna alvo, altere DATASET_PATH ou TARGET_COLUMN em config.py. Por exemplo, para usar uma cópia em data/raw, configure DATASET_PATH como data/raw/Dry_Bean_Dataset.xlsx. Se o arquivo estiver ausente, o programa informa o caminho esperado. O carregador também verifica as características, a coluna alvo, os valores ausentes e inválidos e a quantidade de classes.

## Requisitos

- Python 3.10 ou superior.
- Dependências listadas em requirements.txt.

## Instalação

Na pasta do projeto, crie um ambiente virtual:

    python -m venv .venv

No Windows PowerShell, ative-o e instale as dependências:

    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

No Linux ou macOS:

    source .venv/bin/activate
    python -m pip install -r requirements.txt

## Execução

Depois de colocar o dataset no caminho configurado, execute:

    python main.py

O terminal informa que o dataset foi carregado, mostra o progresso por época e, ao final, apresenta a época selecionada, a acurácia e o relatório de classificação no conjunto de teste. Os parâmetros detalhados ficam registrados em cada execução, mas não são impressos na tela.

O teste é mantido separado durante o treinamento. A validação seleciona a melhor época; somente depois dessa seleção o programa calcula as métricas finais no teste.

## Configuração

Os parâmetros ficam em config.py. Um exemplo para testar uma rede menor por menos épocas é:

    CAMADA_INTERMEDIARIA = [12]
    EPOCAS = 50
    BATCH_SIZE = 64
    TAXA_APRENDIZADO = 0.001

CAMADA_INTERMEDIARIA define dinamicamente a quantidade de camadas ocultas e de neurônios em cada uma. Por exemplo, [16, 8] cria duas camadas ocultas, com dezesseis e oito neurônios. NEURONIOS_ENTRADA deve corresponder às dezesseis características e NEURONIOS_SAIDA às sete classes; valores incompatíveis são rejeitados.

SEED controla os geradores aleatórios da divisão, da inicialização e do embaralhamento, facilitando a reprodução do experimento. EPOCAS define o limite máximo de passagens pelo conjunto de treino. BATCH_SIZE define quantas amostras são processadas em cada atualização. TAXA_APRENDIZADO controla a escala das atualizações dos parâmetros.

HIDDEN_ACTIVATION aceita relu ou sigmoid. OUTPUT_ACTIVATION deve ser softmax para a classificação das sete classes. OPTIMIZER aceita adam ou sgd. LOSS_FUNCTION aceita cross_entropy. L2_LAMBDA maior que zero ativa a penalização L2 dos pesos; zero a desativa. EARLY_STOPPING, PATIENCE e MIN_DELTA controlam a parada antecipada com base na perda de validação. NORMALIZE_DATA ativa a padronização com StandardScaler. SAVE_MODEL, SAVE_PLOTS e SAVE_METRICS controlam as respectivas saídas.

As proporções TRAIN_SIZE, VALIDATION_SIZE e TEST_SIZE precisam somar 1. Com STRATIFIED_SPLIT habilitado, as divisões preservam aproximadamente a proporção das classes. Uma cópia das configurações efetivamente usadas é salva em results/logs a cada execução.

## Como a MLP funciona

Cada amostra fornece dezesseis características à camada de entrada. As camadas intermediárias transformam esses valores e a camada de saída produz uma probabilidade para cada variedade. As ligações entre camadas adjacentes são densas: cada neurônio de uma camada se conecta a todos os neurônios da camada seguinte.

A classe PerceptronMulticamadas, em mlp.py, implementa as operações com NumPy. propaga_para_frente calcula as ativações e guarda os valores intermediários necessários ao treinamento. perda compara as probabilidades com os rótulos corretos usando entropia cruzada. backpropagation calcula os gradientes dos pesos e biases percorrendo as camadas de trás para frente. atualiza_parametros aplica SGD ou Adam aos gradientes. O método treina_lote reúne essas etapas para cada lote. O método backpropagation calcula os gradientes; Adam ou SGD usa esses gradientes para atualizar os parâmetros.

O módulo treinamento.py embaralha as amostras no início de cada época, processa os lotes, registra perda e acurácia de treino e validação e, quando configurado, interrompe o treinamento se a validação deixar de melhorar. Ao encerrar, restaura os parâmetros da melhor época de validação.

## Resultados gerados

Os arquivos ficam dentro de results, organizados por execução:

- results/models/: pesos e biases em NPZ e metadados em JSON, incluindo arquitetura, classes, características e scaler.
- results/metrics/: métricas de teste, relatório de classificação, matriz de confusão e histórico de treino e validação.
- results/plots/: gráficos de perda, acurácia e matriz de confusão.
- results/logs/: cópia JSON da configuração da execução.

A acurácia resume a proporção de previsões corretas. Precisão, recall e F1 ajudam a observar o desempenho por classe. A matriz de confusão mostra quais variedades foram previstas para cada classe real. Os valores dependem da divisão e da configuração da execução; o projeto não fixa uma acurácia esperada.

## Recarregar o modelo e classificar novas amostras

O modelo salvo pode ser carregado sem novo treinamento. Prepare um CSV com uma ou mais novas amostras contendo as dezesseis características nos nomes esperados. O código seleciona e ordena as colunas conforme os metadados do modelo, aplica o scaler salvo, se usado no treinamento, e retorna as classes previstas e suas probabilidades:

    import pandas as pd
    from mlp import PerceptronMulticamadas

    modelo, metadados = PerceptronMulticamadas.carrega(
        "results/models/mlp_<identificador_da_execucao>"
    )
    amostras = pd.read_csv("minhas_amostras.csv")
    X_novo = amostras.loc[:, metadados["feature_names"]].to_numpy(dtype=float)
    probabilidades, classes = modelo.preve_dados_brutos(X_novo, metadados)

    for classe, probabilidades_amostra in zip(classes, probabilidades):
        print(classe, probabilidades_amostra)

Substitua o identificador pelo nome da pasta do modelo. Cada linha do CSV deve conter valores reais para as dezesseis características. A previsão não altera os parâmetros da rede nem executa treinamento.

## Testes

Execute os testes automatizados com:

    python -m pytest -q

Eles verificam o carregamento e a validação dos dados, as dimensões e probabilidades da MLP, a perda e os gradientes, as atualizações, a reprodutibilidade, o aprendizado em um conjunto artificial pequeno e o salvamento e carregamento do modelo. O conjunto artificial é usado somente para testar a implementação; ele não substitui o Dry Bean Dataset na avaliação do projeto.

## Estrutura do projeto

    config.py                  parâmetros do experimento
    main.py                    coordena a execução
    carrega_dataset.py          carrega e valida o arquivo de dados
    preprocessamento.py         divide, codifica e normaliza os dados
    mlp.py  implementa e salva a MLP
    treinamento.py              treina em lotes e aplica parada antecipada
    avaliacao.py                calcula métricas e salva gráficos
    data/raw/                   local alternativo para o arquivo original
    data/processed/             reservado para dados preparados
    results/                    modelos, métricas, gráficos e logs
    artigo_sbc_feijao/          fonte LaTeX e arquivos do modelo SBC
    tests/                      testes automatizados

O arquivo .gitignore impede que o dataset local, os resultados gerados, ambientes virtuais e arquivos temporários sejam enviados ao Git.
