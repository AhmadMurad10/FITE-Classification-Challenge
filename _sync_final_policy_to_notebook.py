import ast
import json
from pathlib import Path


NOTEBOOK = Path("classification_pipeline.ipynb")
PY_FILE = Path("classification_pipeline.py")


def extract_defs(py_text: str, names: list[str]) -> str:
    lines = py_text.splitlines()
    module = ast.parse(py_text)
    found = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = "\n".join(lines[node.lineno - 1:node.end_lineno])
    missing = [name for name in names if name not in found]
    if missing:
        raise RuntimeError(f"Missing definitions from {PY_FILE}: {missing}")
    return "\n\n\n".join(found[name] for name in names)


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
py_text = PY_FILE.read_text(encoding="utf-8")

replacement_groups = {
    "def optimize_ensemble_weights": [
        "optimize_ensemble_weights",
        "select_final_ensemble_probabilities",
        "build_single_model_final_info",
        "build_reference_soft_voting_info",
        "save_reference_soft_voting_candidate",
    ],
    "def greedy_select_ensemble_weights": [
        "greedy_select_ensemble_weights",
        "blend_with_weight_dict",
        "run_nested_greedy_ensemble_audit",
        "save_master_comparison_table",
        "save_model_submission_portfolio",
    ],
    "def main()": ["main"],
}

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))

    source = source.replace(
        'FINAL_SUBMISSION_POLICY = "conservative_private_safe"',
        'FINAL_SUBMISSION_POLICY = "lightgbm_unweighted_original"',
    )

    for marker, names in replacement_groups.items():
        if marker in source:
            source = extract_defs(py_text, names)
            break

    cell["source"] = [line + "\n" for line in source.rstrip().splitlines()]

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
