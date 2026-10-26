"""Executa o notebook real, verifica o resultado e testa a reexecução de serviços.

Uso: python validacao/validar_notebook.py --output-dir /tmp/workshop-validacao
Requer as dependências do notebook e nbclient, nbformat e ipykernel no ambiente.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path(__file__).resolve().parents[1] / "workshop_mlops.ipynb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeat-all", action="store_true", help="Repete todo o laboratório no mesmo kernel")
    args = parser.parse_args()
    output_dir = (args.output_dir or Path(tempfile.mkdtemp(prefix="workshop-validacao-"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, directory in {
        "PREFECT_HOME": "prefect", "MPLCONFIGDIR": "matplotlib",
        "IPYTHONDIR": "ipython", "JUPYTER_RUNTIME_DIR": "jupyter",
    }.items():
        os.environ[key] = str(output_dir / directory)
    nb = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{i}", "exec")
    original_count = len(nb.cells)
    checks_source = Path(__file__).with_name("verificar_resultados.py").read_text()
    checks_source += f"\nPath({str(output_dir / 'resultado.json')!r}).write_text(json.dumps(resumo_validacao, ensure_ascii=False, indent=2))\n"
    extras = [("verificações funcionais", checks_source)]
    for title in ("5. Deploy:", "(Opcional) Abrir a interface"):
        heading = next(i for i, cell in enumerate(nb.cells) if cell.cell_type == "markdown" and title in cell.source)
        source = next(cell.source for cell in nb.cells[heading + 1:] if cell.cell_type == "code")
        extras.append((f"reexecutar {title}", source))
    extras.append(("verificar reexecução dos serviços", '''
assert api_json("GET", "/health")["versao"] == "2"
assert mlflow_ui.poll() is None
assert requests.get(f"{ui_url}/health", timeout=10).ok
print("Reexecução da API e UI: OK")
'''))
    if args.repeat_all:
        # Repete as células originais sem alterar a instalação ou simular dependências.
        extras.extend((f"segunda execução da célula {i}", cell.source)
                      for i, cell in enumerate(nb.cells) if cell.cell_type == "code")
        extras.append(("verificar segunda execução", checks_source))
    extras.append(("encerrar serviços", '''
encerrar_servicos()
assert not api_thread.is_alive()
assert mlflow_ui.poll() is not None
print("Serviços encerrados: OK")
'''))
    for title, source in extras:
        nb.cells.append(nbformat.v4.new_code_cell(source, metadata={"validation_title": title}))

    def before(cell, cell_index, **_):
        if cell.cell_type == "code":
            print(f"EXECUTANDO {cell_index}: {cell.metadata.get('validation_title', cell.source.splitlines()[0])}", flush=True)

    def after(cell, cell_index, **_):
        if cell.cell_type == "code":
            for result in cell.get("outputs", []):
                if result.output_type == "stream":
                    print(result.text[-10000:], flush=True)
                elif result.output_type == "execute_result":
                    print(result.data.get("text/plain", "")[:3000], flush=True)
                elif result.output_type == "error":
                    print(f"{result.ename}: {result.evalue}", flush=True)

    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(output_dir)}},
                            on_cell_start=before, on_cell_executed=after)
    started = time.monotonic()
    success = False
    try:
        with client.setup_kernel():
            # Ensure the kernel uses this interpreter even if another kernelspec is installed.
            nb.cells.insert(0, nbformat.v4.new_code_cell(
                f"import sys; assert sys.executable == {sys.executable!r}, sys.executable"))
            try:
                for i, cell in enumerate(nb.cells):
                    client.execute_cell(cell, i)
                success = True
            finally:
                # Also close the test's own servers when an assertion or notebook cell fails.
                cleanup = nbformat.v4.new_code_cell("if 'encerrar_servicos' in globals(): encerrar_servicos()")
                nb.cells.append(cleanup)
                try:
                    client.execute_cell(cleanup, len(nb.cells) - 1)
                except Exception as exc:
                    print(f"Falha ao encerrar serviços: {exc}", file=sys.stderr)
                nb.cells.pop()
                nb.cells.pop(0)
    finally:
        nbformat.write(nb, output_dir / "execucao_com_testes.ipynb")
        nb.cells = nb.cells[:original_count]
        nbformat.write(nb, output_dir / "workshop_mlops.executado.ipynb")
        if success:
            result_path = output_dir / "resultado.json"
            result = json.loads(result_path.read_text())
            result.update({"status": "aprovado", "segundos": round(time.monotonic() - started, 2),
                           "repeat_all": args.repeat_all,
                           "notebook_sha256": hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
                           "reexecucao_api_ui": "aprovado", "encerramento_servicos": "aprovado"})
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"APROVADO — resultados em {output_dir}")


if __name__ == "__main__":
    main()
