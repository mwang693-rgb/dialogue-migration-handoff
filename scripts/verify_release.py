"""Check public handoff integrity and a bounded secret scan; no network/GPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'provenance'/'release_manifest.json'
EXCLUDED = {'.git', '__pycache__', '.venv', 'runs', 'models', 'dist'}
PATTERNS = [
    re.compile(r'(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{16,}'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(r'(?im)^\s*(?:password|passwd|api_key|access_token)\s*[:=]\s*[\'"][^\'"\r\n]{8,}[\'"]'),
    re.compile(r'connect\.[A-Za-z0-9.-]*seetacloud\.com'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and p != MANIFEST and not set(p.relative_to(ROOT).parts).intersection(EXCLUDED))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', action='store_true', help='Only for maintainers before a new release')
    args = parser.parse_args()
    files = public_files()
    for path in files:
        assert path.stat().st_size < 10_000_000, f'Unexpected large file: {path.name}'
        text = path.read_text(encoding='utf-8-sig')
        assert not any(p.search(text) for p in PATTERNS), f'Potential credential/server address in {path.relative_to(ROOT)}; value intentionally not printed.'
        if path.suffix == '.py':
            ast.parse(text, filename=str(path.relative_to(ROOT)))
    originals = json.loads((ROOT/'provenance/original_files.json').read_text(encoding='utf-8'))
    for row in originals:
        if row.get('exact_copy'):
            assert sha(ROOT/row['destination']) == row['original_file_sha256'], row['destination']
    current = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    if args.create:
        assert not MANIFEST.exists(), 'Do not silently overwrite a release manifest.'
        MANIFEST.write_text(json.dumps(dict(files=current, original_files_verified=len(originals), scan_scope='Known token/private-key/credential assignment and former server host patterns; not a guarantee against every possible secret.'), ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    else:
        expected = json.loads(MANIFEST.read_text(encoding='utf-8'))['files']
        assert current == expected, 'Release file list/hash mismatch; do not publish unreviewed changes.'
    print(json.dumps(dict(verified=True, files=len(current), bytes=sum(p.stat().st_size for p in files), syntax_checked=True, original_exact_copies_checked=True, credential_pattern_matches=0)))


if __name__ == '__main__':
    main()
