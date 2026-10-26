"""Executado pelo validador dentro do kernel, depois das células do laboratório."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

verificacoes = []


def conferir(nome, condicao):
    assert condicao, nome
    verificacoes.append(nome)


conferir("quality gate rejeita candidato idêntico", resultado_gate is None)
conferir("retreino promove nova versão", str(versao_retreino) == "2")
conferir("retreino melhora no teste reservado", acc_depois > acc_antes)
conferir("API usa o champion do Registry",
         api_json("GET", "/health")["versao"] == str(client.get_model_version_by_alias(MODEL_NAME, "champion").version))

grupos = [set(row.tobytes() for row in matriz) for matriz in (X_train, X_val, X_test)]
conferir("treino, validação e teste têm imagens distintas",
         all(not grupos[i].intersection(grupos[j]) for i, j in ((0, 1), (0, 2), (1, 2))))
conferir("hash estável para cópias dos dados",
         versao_dados(X_train, y_train) == versao_dados(X_train.copy(), y_train.copy()))
conferir("câmera simulada é determinística", np.array_equal(camera_nova(X_test), camera_nova(X_test)))

invalidos = [
    ("matriz unidimensional", X_train[0], y_train),
    ("matriz vazia", X_train[:0], y_train[:0]),
    ("quantidade de rótulos", X_train, y_train[:-1]),
    ("classe ausente", X_train[y_train != 9], y_train[y_train != 9]),
    ("classe inválida", X_train, np.full_like(y_train, 10)),
]
for valor, nome in [(np.nan, "NaN"), (np.inf, "infinito"), (-1, "pixel negativo"), (17, "pixel acima do limite")]:
    matriz = X_train.copy()
    matriz[0, 0] = valor
    invalidos.append((nome, matriz, y_train))
for nome, matriz, rotulos in invalidos:
    try:
        validar_dados(matriz, rotulos)
    except ValueError:
        verificacoes.append("validação rejeita " + nome)
    else:
        raise AssertionError("Validação aceitou " + nome)

conferir("sem alerta no tráfego original", not checar_drift(X_train, X_prod_ok[:100])[0])
conferir("alerta no tráfego da nova câmera", checar_drift(X_train, X_prod_drift[-100:])[0])
try:
    checar_drift(X_train, np.empty((0, 64)))
except ValueError:
    verificacoes.append("drift rejeita janela vazia")
else:
    raise AssertionError("Drift aceitou janela vazia")

with sqlite3.connect(DB) as con:
    total_antes = con.execute("SELECT COUNT(*) FROM predicoes").fetchone()[0]
payloads_invalidos = [
    {}, {"pixels": [0] * 63}, {"pixels": [0] * 65},
    {"pixels": [-1] + [0] * 63}, {"pixels": [17] + [0] * 63},
    {"pixels": [None] + [0] * 63}, {"pixels": ["NaN"] + [0] * 63},
    {"pixels": [0] * 64, "campo_desconhecido": True},
]
for indice, payload in enumerate(payloads_invalidos):
    resposta = requests.post(f"{API}/predict", json=payload, timeout=10)
    conferir(f"API rejeita payload inválido {indice + 1} com HTTP 422", resposta.status_code == 422)
with sqlite3.connect(DB) as con:
    conferir("entradas inválidas não entram no log",
             con.execute("SELECT COUNT(*) FROM predicoes").fetchone()[0] == total_antes)

def requisicao_concorrente(indice):
    if indice % 4 == 0:
        api_json("PUT", "/reload")
    return api_json("POST", "/predict", json={"pixels": X_test[indice].tolist()})

with ThreadPoolExecutor(max_workers=4) as pool:
    respostas = list(pool.map(requisicao_concorrente, range(16)))
conferir("predições concorrentes durante reload", all(r["versao_modelo"] == "2" for r in respostas))
with sqlite3.connect(DB) as con:
    linhas = con.execute("SELECT entrada, predicao, versao FROM predicoes ORDER BY rowid DESC LIMIT 16").fetchall()
for entrada, predicao, versao in linhas:
    modelo = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{versao}")
    conferir("log vincula predição à versão correta", int(modelo.predict([json.loads(entrada)])[0]) == predicao)

@flow(name="verificar-quality-gate")
def verificar_gate():
    pior_run = runs.iloc[-1].run_id
    conferir("quality gate rejeita candidato inferior", not t_quality_gate(pior_run, X_val, y_val))
    run_atual = client.get_model_version_by_alias(MODEL_NAME, "champion").run_id
    X_ev = np.vstack([X_val, X_novo_ev])
    y_ev = np.concatenate([y_val, y_novo_ev])
    conferir("quality gate aplica mínimo de 99%", not t_quality_gate(run_atual, X_ev, y_ev, minimo=0.99))
    # Uma falha de infraestrutura precisa interromper a avaliação.
    with patch.object(mlflow.sklearn, "load_model", side_effect=[
        estado["modelo"], MlflowException("falha simulada", error_code="INTERNAL_ERROR")
    ]):
        try:
            t_quality_gate.fn(run_atual, X_val, y_val)
        except MlflowException:
            verificacoes.append("falha do Registry não aprova o candidato")
        else:
            raise AssertionError("Falha de infraestrutura foi ignorada")

verificar_gate()
enviar_trafego(X_prod_ok[:100])
quantidade_runs = len(mlflow.search_runs(experiment_ids=[EXPERIMENT_ID]))
conferir("monitor não retreina sem drift", monitorar_e_retreinar() == "sem drift — nada a fazer")
conferir("monitor sem drift não cria run de treino",
         len(mlflow.search_runs(experiment_ids=[EXPERIMENT_ID])) == quantidade_runs)

resposta = requests.get(f"{ui_url}/api/2.0/mlflow/experiments/get",
                        params={"experiment_id": EXPERIMENT_ID}, timeout=10)
resposta.raise_for_status()
conferir("UI consulta os mesmos artefatos do notebook",
         resposta.json()["experiment"]["artifact_location"] == client.get_experiment(EXPERIMENT_ID).artifact_location)

resumo_validacao = {
    "python": sys.version.split()[0],
    "plataforma": __import__("platform").platform(),
    "bibliotecas": {p: version(p) for p in (
        "mlflow", "prefect", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "fastapi", "uvicorn"
    )},
    "amostras": {"treino": len(X_train), "validacao": len(X_val), "teste": len(X_test)},
    "accuracy_teste_drift_antes": acc_antes,
    "accuracy_teste_drift_depois": acc_depois,
    "champion": str(client.get_model_version_by_alias(MODEL_NAME, "champion").version),
    "verificacoes": verificacoes,
}
print(json.dumps(resumo_validacao, ensure_ascii=False, indent=2))
