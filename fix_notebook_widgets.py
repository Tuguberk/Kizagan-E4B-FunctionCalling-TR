#!/usr/bin/env python3
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('turkish_llm_finetune.ipynb')
if not p.exists():
    print('Dosya bulunamadı:', p)
    sys.exit(1)

nb = json.loads(p.read_text())
modified = 0

def ensure_state_on_widgets(d):
    global modified
    if not isinstance(d, dict):
        return
    if 'metadata' in d and isinstance(d['metadata'], dict) and 'widgets' in d['metadata']:
        w = d['metadata']['widgets']
        if isinstance(w, dict) and 'state' not in w:
            w['state'] = {}
            d['metadata']['widgets'] = w
            modified += 1
    # Recurse into values
    for v in d.values():
        if isinstance(v, dict):
            ensure_state_on_widgets(v)
        elif isinstance(v, list):
            for item in v:
                ensure_state_on_widgets(item)

# Top-level
ensure_state_on_widgets(nb)

if modified > 0:
    backup = p.with_suffix(p.suffix + '.bak')
    backup.write_text(p.read_text())
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=2))
    print(f'Güncellendi: {modified} adet metadata.widgets öğesine "state": {{}} eklendi. Yedek: {backup}')
else:
    print('Değişiklik yapılmadı (metadata.widgets eksik veya zaten state mevcut).')
