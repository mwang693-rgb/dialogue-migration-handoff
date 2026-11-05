"""Content editing checks distinguish serialization wrappers and docstring whitespace."""
import ast
import copy
import inspect
import json
from run_artifact_expansion import followups, ast_signature, check as old_check


def unwrap(text):
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def parse(text, family):
    from run_artifact_expansion import parse_artifact
    return parse_artifact(unwrap(text), family)


def normalized_ast(tree, query_id):
    result = copy.deepcopy(tree)
    if query_id == "set_docstring":
        for n in result.body:
            if isinstance(n, ast.FunctionDef) and n.body and isinstance(n.body[0], ast.Expr):
                v = n.body[0].value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    v.value = inspect.cleandoc(v.value)
    return ast_signature(result)


def check(text, family, q):
    raw = text.strip()
    cleaned = unwrap(raw)
    wrapper = dict(has_fence=raw.startswith("```"), closed_fence=raw.endswith("```") if raw.startswith("```") else None)
    if family != "code":
        result = old_check(cleaned, family, q)
        result["serialization_wrapper"] = wrapper
        return result
    try:
        tree = ast.parse(cleaned)
        strict = ast_signature(tree) == q["expected_ast"]
        # Expected AST is produced from actual prior code plus the specified edit.
        # A string dump cannot be re-parsed as Python; construct the expected
        # docstring normalization through a separate expected code supplied by q.
        if q["query_id"] == "set_docstring":
            expected_tree = ast.parse(q["expected_code"])
            passed = normalized_ast(tree, q["query_id"]) == normalized_ast(expected_tree, q["query_id"])
        else:
            passed = strict
        return dict(all_pass=passed, strict_ast_pass=strict, valid_syntax=True, serialization_wrapper=wrapper,
            note="Docstring-only whitespace normalization for docstring edits; executable AST unchanged; no code execution")
    except (ValueError, SyntaxError, TypeError) as exc:
        return dict(all_pass=False, parse_error=str(exc), serialization_wrapper=wrapper)


def make_queries(spec, value):
    qs = followups(spec, value)
    if spec["family"] == "code":
        # Supply actual expected Python rather than eval-ing any AST dump.
        tree = copy.deepcopy(value["tree"])
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        doc = ast.Expr(value=ast.Constant(value=f"Working implementation version {spec['variant'] + 1}."))
        if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str):
            fn.body[0] = doc
        else:
            fn.body.insert(0, doc)
        ast.fix_missing_locations(tree)
        qs[1]["expected_code"] = ast.unparse(tree)
    return qs
