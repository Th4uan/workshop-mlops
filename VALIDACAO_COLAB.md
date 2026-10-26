# Validação do workshop de MLOps

Data: 26/10/2026. Arquivo revisado: `workshop_mlops.ipynb`.

**Resultado:** execução local aprovada em um kernel Jupyter real, com Python 3.12.14, CPU, NumPy 2.0.2 e pandas 2.2.2. Todas as 15 células de código foram executadas, incluindo a instalação e a UI opcional. O laboratório foi repetido no mesmo kernel, com novos bancos e artefatos, e passou novamente.

Foram aprovadas 51 verificações por execução, incluindo 16 conferências de predições registradas. A validação completa, com duas execuções e os testes adicionais, levou 72,28 segundos depois da preparação do ambiente. Instalação inicial, criação do cache de fontes e inicialização dos serviços podem aumentar esse tempo.

**Limite da validação:** a execução aconteceu em macOS ARM64, não em uma VM Linux hospedada pelo Colab. O proxy de `google.colab.output.serve_kernel_port_as_window` e possíveis conflitos com outras bibliotecas preinstaladas do Colab ainda precisam de uma conferência na sessão real. Python 3.12 e NumPy 2.0.2 constam nos [runtimes documentados pelo Colab](https://research.google.com/colaboratory/runtime-version-faq.html).

Antes da publicação, o endereço informado do GitHub retornou 404 tanto pelo acesso público quanto pelo conector autenticado. A revisão e os testes foram feitos sobre a cópia local. A versão corrigida foi preparada para publicação em `https://github.com/Th4uan/workshop-mlops`, com o mesmo conteúdo de notebook identificado pelo hash em `validacao/resultado.json`.

## Correções aplicadas

- Instalação pelo Python do kernel, com interrupção explícita se o pip falhar e declaração de todas as dependências diretas.
- Prazo de inicialização do Prefect ampliado de 20 para 120 segundos: o prazo original foi ultrapassado em uma execução de teste.
- Separação em treino, validação e teste. O original reutilizava imagens do treino, transformadas, na avaliação do retreino. Agora nenhuma imagem do teste entra no treino ou no quality gate.
- Transformação da câmera determinística e texto ajustado para imagens mais claras/lavadas.
- Pastas independentes para cada execução, com caminhos absolutos para os bancos e artefatos.
- Uso do URI retornado pelo MLflow 3 ao registrar os modelos, evitando a compatibilidade indireta com caminhos antigos de artefatos.
- Cache das tasks desativado para executar novamente operações com efeitos externos.
- Falhas do Registry propagadas pelo quality gate; somente a ausência do champion recebe tratamento específico.
- API com validação dos 64 pixels, valores finitos e intervalo permitido; troca conjunta do modelo e de sua versão durante o reload.
- Chamadas HTTP com timeout e checagem de status; inicialização da API com verificação de prontidão.
- UI sem comando de shell, usando porta livre, processo reutilizável e verificação de prontidão. API e UI podem ser encerradas com `encerrar_servicos()`.

## Resultados observados

| Medida | Resultado |
|---|---:|
| Imagens de treino / validação / teste | 1.078 / 359 / 360 |
| Acurácia de validação do champion inicial, cenário limpo | 98,3% |
| Acurácia no tráfego de teste limpo | 99,3% |
| Acurácia na nova câmera, antes do retreino | 78,1% |
| Acurácia na mesma nova câmera, após o retreino | 86,7% |
| Acurácia de validação do candidato, cenários limpo + novo | 93,5% |
| Versão promovida e carregada pela API | 2 |

As métricas de validação das versões 1 e 2 usam cenários diferentes; não devem ser comparadas diretamente na tabela de auditoria. O quality gate calcula candidato e champion sobre o mesmo conjunto a cada decisão. Os percentuais de teste medem esta simulação e não constituem uma estimativa geral de desempenho em produção.

Também foram verificados: rejeição de candidatos iguais ou inferiores, mínimo de 99%, erros do Registry, dados malformados, oito tipos de payload inválido, ausência de logs para entradas rejeitadas, predições concorrentes durante reload, correspondência entre predição e versão registrada, drift presente/ausente, ausência de retreino sem drift, leitura dos mesmos artefatos pela UI e encerramento dos serviços.

## Arquivos e reprodução

- `workshop_mlops.ipynb`: notebook corrigido, sem outputs, para distribuir.
- `validacao/workshop_mlops.executado.ipynb`: primeira execução completa, com gráficos e resultados.
- `validacao/execucao_com_testes.ipynb`: evidência das duas execuções e verificações adicionais.
- `validacao/resultado.json`: resultado estruturado, versões e hash SHA-256 do notebook testado.
- `validacao/workshop_mlops.original.ipynb`: cópia exata do arquivo anterior às alterações.
- `validacao/requirements-testados.txt`: versões diretas usadas na validação.

Para repetir localmente com um ambiente isolado:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r validacao/requirements-testados.txt
.venv/bin/python validacao/validar_notebook.py --output-dir /tmp/workshop-validacao --repeat-all
```

É necessário permitir portas locais para Jupyter, Prefect, API e MLflow. O validador executa o notebook real e encerra os serviços iniciados por ele.

Para a conferência final no Colab, envie `workshop_mlops.ipynb` por **Arquivo → Fazer upload de notebook**, selecione **Python 3 / CPU** e use **Executar tudo** em uma sessão nova. Confira a promoção para v2, a melhora de 78,1% para 86,7% nesta simulação e o link da UI opcional. Os arquivos da sessão são temporários, conforme a [documentação do Colab](https://research.google.com/colaboratory/faq.html).
