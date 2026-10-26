# Workshop: MLOps na prática

Um laboratório para percorrer o ciclo de vida de um modelo de machine learning em um notebook: validação dos dados, treino, rastreamento, registro, API de inferência, monitoramento de drift e retreino com quality gate.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Th4uan/workshop-mlops/blob/main/workshop_mlops.ipynb)

## Executar no Google Colab

1. Abra o notebook pelo botão acima.
2. Selecione um ambiente **Python 3 / CPU**. Não é necessário GPU, token ou Google Drive.
3. Em uma sessão nova, use **Ambiente de execução → Executar tudo** e aguarde a instalação e a inicialização dos serviços.
4. Ao terminar, execute `encerrar_servicos()` em uma nova célula para parar a API e a UI do MLflow.

O dataset Digits acompanha o scikit-learn. Os bancos SQLite e os modelos são criados em uma pasta temporária por execução; eles não são incluídos ao baixar o notebook.

## O que o laboratório demonstra

- Hash dos dados e validações antes do treino.
- Comparação de três classificadores e rastreamento com MLflow.
- Model Registry com alias `champion`.
- Pipeline Prefect com aprovação do candidato antes da promoção.
- FastAPI para predição, logs em SQLite e reload do modelo.
- Mudança simulada da câmera, detecção de drift e retreino com dados rotulados simulados.
- Avaliação com conjuntos separados de treino, validação e teste.

## Validação

O notebook passou por **duas execuções completas no mesmo kernel Jupyter**, com **51 verificações aprovadas por execução**, em um ambiente local com Python 3.12.14 e CPU. Na simulação da nova câmera, a acurácia no teste reservado passou de **78,1% para 86,7%** após o retreino.

A execução em uma VM hospedada pelo Colab e o acesso à UI pelo proxy visual do Google ainda precisam ser conferidos. As versões diretas usadas no teste estão em [`validacao/requirements-testados.txt`](validacao/requirements-testados.txt).

- [Relatório completo](VALIDACAO_COLAB.md)
- [Notebook executado com gráficos e resultados](validacao/workshop_mlops.executado.ipynb)
- [Evidência das execuções e dos testes adicionais](validacao/execucao_com_testes.ipynb)
- [Resultado estruturado e hash do notebook testado](validacao/resultado.json)

Para reproduzir os testes localmente, com Python 3.12 instalado:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r validacao/requirements-testados.txt
.venv/bin/python validacao/validar_notebook.py --output-dir /tmp/workshop-validacao --repeat-all
```

O ambiente precisa permitir portas locais para Jupyter, Prefect, API e MLflow.

## Referência

O laboratório foi inspirado no projeto [full-stack-on-prem-cv-mlops](https://github.com/jomariya23156/full-stack-on-prem-cv-mlops), adaptando as etapas para uma sessão de notebook.
