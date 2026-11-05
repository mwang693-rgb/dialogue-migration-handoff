"""Followups target actual Qwen artifacts, with retention and edit checks; no code execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch
from run_pilot import Runner, append_jsonl, digest, messages, read_jsonl, write_json

OUT = Path("results/artifact_expansion_v1")


def unfence(s):
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        s = "\n".join(s.splitlines()[1:-1])
    return s


def parse_artifact(text, family):
    s = unfence(text)
    if family == "code":
        tree = ast.parse(s)
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert len(fns) == 1, "Expected one top-level function"
        return dict(code=s, name=fns[0].name, tree=tree)
    value = json.loads(s)
    assert isinstance(value, dict)
    if family == "book":
        assert set(value) == {"pages"} and len(value["pages"]) == 8
        assert all(isinstance(x, str) and x.strip() for x in value["pages"])
        assert len(set(value["pages"])) == 8, "Distinct pages needed for unambiguous retention checks"
    else:
        assert set(value) == {"items"} and len(value["items"]) == 6
        for x in value["items"]:
            assert isinstance(x, dict) and set(x) == {"id", "location", "budget", "status"}
            assert all(isinstance(x[k], str) and x[k].strip() for k in ["id", "location", "status"])
            assert type(x["budget"]) is int
        assert len({x["id"] for x in value["items"]}) == 6
    return value


def renamed(tree, old, new):
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == old:
                node.name = new
            return self.generic_visit(node)
        def visit_Name(self, node):
            if node.id == old:
                node.id = new
            return node
    return Rename().visit(copy.deepcopy(tree))


def ast_signature(tree):
    return ast.dump(tree, include_attributes=False)


def followups(spec, value):
    family, i = spec["family"], spec["variant"]
    if family == "book":
        index = [5, 1, 6, 3][i]
        replacement = f"At sunrise, the friends celebrated their small victory beside a golden tree {i + 1}."
        e1 = copy.deepcopy(value)
        e1["pages"][index] = replacement
        e2 = copy.deepcopy(value)
        e2["pages"][:3] = list(reversed(e2["pages"][:3]))
        return [dict(query_id="replace_page", text=f"In our latest book, replace page {index + 1} with exactly: {replacement}\nReturn the full eight-page JSON book. Keep every other page string exactly unchanged.", expected=e1,
                     original=value, changed_indices=[index], expected_retained=7),
                dict(query_id="reorder_pages", text="In our latest book, reverse the order of only the first three pages. Keep all eight page strings exactly unchanged. Return the complete JSON book.", expected=e2,
                     original=value, changed_indices=[], expected_retained=8)]
    if family == "table":
        index = [1, 4, 2, 0][i]
        row_id = value["items"][index]["id"]
        e1 = copy.deepcopy(value)
        e1["items"].pop(index)
        e2 = copy.deepcopy(value)
        e2["items"][index]["budget"] += 7
        new_budget = e2["items"][index]["budget"]
        return [dict(query_id="delete_row", text=f"In our latest plan, delete only the item with id {json.dumps(row_id)}. Return the full remaining five-item JSON plan. Preserve the order and every field of all remaining items.", expected=e1,
                     original=value, changed_indices=[index], expected_retained=5),
                dict(query_id="update_budget", text=f"In our latest plan, change only the budget of item {json.dumps(row_id)} to {new_budget}. Return the complete six-item JSON plan. Keep all other fields and item order exactly unchanged.", expected=e2,
                     original=value, changed_indices=[index], expected_retained=5)]
    old = value["name"]
    new = f"working_function_final_{i}"
    assert new not in {n.id for n in ast.walk(value["tree"]) if isinstance(n, ast.Name)}
    e1 = renamed(value["tree"], old, new)
    e2 = copy.deepcopy(value["tree"])
    fn = next(n for n in e2.body if isinstance(n, ast.FunctionDef))
    description = f"Working implementation version {i + 1}."
    doc = ast.Expr(value=ast.Constant(value=description))
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str):
        fn.body[0] = doc
    else:
        fn.body.insert(0, doc)
    return [dict(query_id="rename_function", text=f"In our latest implementation, rename function {old} to {new} and update references to it. Return the complete Python code. Keep every other executable statement unchanged; no examples or tests.", expected_ast=ast_signature(e1)),
            dict(query_id="set_docstring", text=f"In our latest implementation, set the function's docstring to exactly {json.dumps(description)}. Replace its old docstring if present. Keep the function name and every executable statement unchanged. Return the complete Python code only.", expected_ast=ast_signature(e2))]


def check(text, family, q):
    try:
        s = unfence(text)
        if family == "code":
            tree = ast.parse(s)
            return dict(all_pass=ast_signature(tree) == q["expected_ast"], valid_syntax=True,
                        note="AST equality for specified edit; generated code never executed and algorithm correctness untested")
        actual = json.loads(s)
        key = "pages" if family == "book" else "items"
        got = actual.get(key, []) if isinstance(actual, dict) else []
        expected = q["expected"][key]
        schema = isinstance(actual, dict) and set(actual) == {key} and isinstance(got, list)
        if family == "book":
            schema = schema and all(isinstance(x, str) for x in got)
        else:
            schema = schema and all(isinstance(x, dict) and set(x) == {"id", "location", "budget", "status"} and type(x["budget"]) is int and all(isinstance(x[k], str) for k in ["id", "location", "status"]) for x in got)
        retained = sum(x in got for index, x in enumerate(q["original"][key]) if index not in q["changed_indices"])
        return dict(all_pass=actual == q["expected"], valid_schema=bool(schema), actual_items=len(got),
                    expected_items=len(expected), retained_unchanged_items=retained, expected_retained=q["expected_retained"])
    except (ValueError, SyntaxError, TypeError, AttributeError) as exc:
        return dict(all_pass=False, parse_error=str(exc))


def main():
    specs_path = Path("data/data_expansion_v1/artifact_specs.jsonl")
    specs = read_jsonl(specs_path)
    assert len(specs) == 12
    assert not OUT.exists(), "Keep partial runs; use a new directory for repairs"
    OUT.mkdir(parents=True)
    config = dict(model="Qwen2.5-7B-Instruct", dtype="bfloat16", seed=453, do_sample=False, history_cap=1536, answer_cap=1024, max_input=8192,
        specs_sha256=hashlib.sha256(specs_path.read_bytes()).hexdigest(),
        code_sha256={n:hashlib.sha256(Path("scripts", n).read_bytes()).hexdigest() for n in ["run_artifact_expansion.py", "run_pilot.py", "pilot_data.py", "build_data_expansion.py"]},
        query_generation="deterministic followups derived from actual second Qwen artifact; both queries branch from same frozen history",
        construction_failures="retain all attempted instances; no retries or outcome filtering", code_execution=False,
        unit="12 designed instances sharing 3 families, 24 paired queries; not 24 independent dialogues")
    ch = digest(config)
    write_json(OUT / "config.json", config)
    torch.manual_seed(453)
    runner = Runner("/root/autodl-tmp/models/qwen2.5-7b-instruct", 15, 8192)
    for index, spec in enumerate(specs):
        print(json.dumps(dict(case_id=spec["case_id"], completed=index, total=12)), flush=True)
        write_json(Path("artifacts/artifact_expansion_status.json"), dict(phase="running", case_id=spec["case_id"], completed=index, total=12, updated_unix=time.time()))
        family = spec["family"]
        hist = [dict(role="user", content=spec["prompt"])]
        try:
            a1 = runner.generate(messages(hist), 1536)
            append_jsonl(OUT / "construction_steps.jsonl", dict(case_id=spec["case_id"], turn=1, config_hash=ch, **a1))
            assert not a1["truncated"], "First artifact reply capped"
            v1 = parse_artifact(a1["text"], family)
            hist.append(dict(role="assistant", content=a1["text"]))
            if family == "book":
                edit = "The friends rested beneath a bright blue umbrella."
                user2 = f"Revise this book: replace page 1 with exactly {json.dumps(edit)}. Keep the other seven page strings exactly unchanged. Return the complete eight-page JSON book only."
                prior_expected = copy.deepcopy(v1)
                prior_expected["pages"][0] = edit
            elif family == "table":
                row_id = v1["items"][-1]["id"]
                user2 = f"Revise this plan: change only the status of item {json.dumps(row_id)} to confirmed. Keep every other field unchanged and return the complete six-item JSON plan only."
                prior_expected = copy.deepcopy(v1)
                prior_expected["items"][-1]["status"] = "confirmed"
            else:
                new_name = f"working_function_{spec['variant']}"
                user2 = f"Rename function {v1['name']} to {new_name} and update references to it. Keep every other executable statement unchanged. Return the complete Python code only."
                prior_expected = ast_signature(renamed(v1["tree"], v1["name"], new_name))
            hist.append(dict(role="user", content=user2))
            a2 = runner.generate(messages(hist), 1536)
            append_jsonl(OUT / "construction_steps.jsonl", dict(case_id=spec["case_id"], turn=2, config_hash=ch, **a2))
            assert not a2["truncated"], "Second artifact reply capped"
            v2 = parse_artifact(a2["text"], family)
            hist.append(dict(role="assistant", content=a2["text"]))
            prior_pass = ast_signature(v2["tree"]) == prior_expected if family == "code" else v2 == prior_expected
            queries = followups(spec, v2)
            serial_gold = {k:v for k,v in v2.items() if k != "tree"} if family == "code" else v2
            append_jsonl(OUT / "histories.jsonl", dict(case_id=spec["case_id"], family=family, config_hash=ch,
                history=hist, latest_artifact=serial_gold, prior_requested_edit_passed=prior_pass, queries=queries, history_truncated=False))
            for q in queries:
                for fmt in ["N", "P"]:
                    a = runner.generate(messages(hist, q["text"], pasted=fmt == "P"), 1024)
                    append_jsonl(OUT / "outputs.jsonl", dict(case_id=spec["case_id"], family=family, query_id=q["query_id"],
                        format=fmt, config_hash=ch, query=q["text"], checker=check(a["text"], family, q), **a))
        except Exception as exc:
            append_jsonl(OUT / "errors.jsonl", dict(case_id=spec["case_id"], family=family, config_hash=ch,
                error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), partial_history=hist))
            print("CONSTRUCTION/CASE ERROR", spec["case_id"], str(exc), flush=True)
    summary = dict(phase="finished", attempted_instances=len(specs), constructed=len(read_jsonl(OUT / "histories.jsonl")),
        outputs=len(read_jsonl(OUT / "outputs.jsonl")), errors=read_jsonl(OUT / "errors.jsonl"), config_hash=ch)
    write_json(OUT / "summary.json", summary)
    write_json(Path("artifacts/artifact_expansion_status.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
