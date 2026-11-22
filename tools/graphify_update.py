"""Rebuild the graphify code graph of the repository (graphify-out/).

Structural (AST) extraction only: no LLM and no tokens; unchanged files are served
from graphify-out/cache. The whole graph is rebuilt on every run, so renamed or
deleted files leave no stale nodes. Run it after any change to the project structure
and commit graphify-out/ together with that change:

    %USERPROFILE%\\.graphify-venv\\Scripts\\python.exe tools/graphify_update.py
"""
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "graphify-out"
EXCLUDES = ["U-CUDA/library", "third_party", ".claude", ".codex", "vcpkg_installed", "x64",
            ".venv", "tests/baselines", "tests/fixtures/library", ".vscode", "graphify-out"]


def module_of(source_file):
    parts = PurePosixPath(str(source_file).replace("\\", "/")).parts
    if not parts:
        return "?"
    if parts[0] in ("libs", "apps") and len(parts) > 1:
        return parts[1]
    return parts[0]


def community_labels(G, communities):
    """Name = dominant module + the most connected node of the community."""
    labels, used = {}, collections.Counter()
    for cid, members in communities.items():
        members = [n for n in members if n in G]
        mods = collections.Counter(module_of(G.nodes[n].get("source_file", "")) for n in members)
        mod = mods.most_common(1)[0][0] if mods else "?"
        top = max(members, key=G.degree, default=None)
        name = G.nodes[top].get("label", str(top)).strip("()") if top else ""
        label = f"{mod}: {name}" if name else mod
        used[label] += 1
        labels[cid] = label if used[label] == 1 else f"{label} #{used[label]}"
    return labels


def main():
    from graphify.analyze import god_nodes, suggest_questions, surprising_connections
    from graphify.build import build_from_json
    from graphify.cli import _stamped_manifest_files
    from graphify.cluster import cluster, score_all
    from graphify.detect import detect, save_manifest
    from graphify.export import to_json
    from graphify.extract import collect_files, extract
    from graphify.report import generate

    OUT.mkdir(exist_ok=True)
    (OUT / ".graphify_build.json").write_text(
        json.dumps({"excludes": EXCLUDES, "gitignore": True}, indent=2), encoding="utf-8")
    (OUT / ".graphify_root").write_text(str(ROOT), encoding="utf-8")
    (OUT / ".graphify_python").write_text(sys.executable, encoding="utf-8")

    found = detect(ROOT, extra_excludes=EXCLUDES, gitignore=True)
    code_files = []
    for f in found["files"].get("code", []):
        code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])
    ast = extract(code_files, cache_root=ROOT, root=ROOT)
    extraction = {"nodes": ast["nodes"], "edges": ast["edges"], "hyperedges": [],
                  "input_tokens": 0, "output_tokens": 0}

    G = build_from_json(extraction, root=str(ROOT), directed=False)
    if G.number_of_nodes() == 0:
        raise SystemExit("graphify: empty graph, nothing written")
    communities = cluster(G)
    cohesion = score_all(G, communities)
    labels = community_labels(G, communities)
    # force: a full rebuild legitimately shrinks the graph when files are deleted.
    to_json(G, communities, str(OUT / "graph.json"), force=True, community_labels=labels)
    report = generate(G, communities, cohesion, labels, god_nodes(G),
                      surprising_connections(G, communities), found, {"input": 0, "output": 0},
                      str(ROOT), suggested_questions=suggest_questions(G, communities, labels))
    (OUT / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / ".graphify_labels.json").write_text(
        json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False, indent=0), encoding="utf-8")

    corpus = found["files"]
    (OUT / "manifest.json").unlink(missing_ok=True)
    save_manifest(_stamped_manifest_files(corpus, extraction, ROOT, partial_source_files=set()),
                  root=str(ROOT), scan_corpus={f for fl in corpus.values() for f in fl})

    cost_path = OUT / "cost.json"
    cost = (json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists()
            else {"runs": [], "total_input_tokens": 0, "total_output_tokens": 0})
    cost["runs"].append({"date": datetime.now(timezone.utc).isoformat(),
                         "input_tokens": 0, "output_tokens": 0, "files": found["total_files"]})
    cost_path.write_text(json.dumps(cost, indent=2), encoding="utf-8")

    import subprocess
    exe = Path(sys.executable).with_name("graphify.exe" if sys.platform == "win32" else "graphify")
    subprocess.run([str(exe), "export", "html"], cwd=ROOT, check=True)
    print(f"graphify: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(communities)} communities, {len(code_files)} code files")


if __name__ == "__main__":
    main()
