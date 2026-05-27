import json
import os
import requests
import tempfile
from getpass import getpass
from huggingface_hub import HfApi

REPOS = [
    "Tuguberk/Kizagan-E4B-Turkish-Agent-FunctionCalling-Hermes",
    "Tuguberk/Kizagan-E4B-Turkish-Agent-FunctionCalling-Hermes-lora",
]

NEW_CHAT_TEMPLATE = """{{ bos_token }}
{%- set tools_str = tools | tojson(indent=2) if tools else "" %}
{%- set tool_system = "" %}
{%- if tools %}
{%- set tool_system = "Fonksiyon çağırma yeteneğine sahip bir yapay zeka modelisiniz. Size <tools> </tools> XML etiketleri içinde fonksiyon imzaları sağlanmıştır. Kullanıcı sorgusuna yardımcı olmak için bir veya daha fazla fonksiyonu çağırabilirsiniz. Fonksiyonlara hangi değerlerin girileceği konusunda varsayımlarda bulunmayın.\\n<tools>\\n" + tools_str + "\\n</tools>\\nHer fonksiyon çağrısı için, aşağıdaki şema ile <tool_call> </tool_call> etiketleri içinde fonksiyon adı ve argümanları içeren bir json nesnesi döndürün:\\n<tool_call>\\n{\\"name\\": <fonksiyon-adı>, \\"arguments\\": <args-sözlüğü>}\\n</tool_call>" %}
{%- endif %}
{%- set first_user_extra = "" %}
{%- set loop_messages = messages %}
{%- if messages[0]['role'] == 'system' %}
{%- if tool_system %}
{%- set first_user_extra = messages[0]['content'] + "\\n\\n" + tool_system + "\\n\\n" %}
{%- else %}
{%- set first_user_extra = messages[0]['content'] + "\\n\\n" %}
{%- endif %}
{%- set loop_messages = messages[1:] %}
{%- elif tool_system %}
{%- set first_user_extra = tool_system + "\\n\\n" %}
{%- endif %}
{%- for message in loop_messages %}
{%- if message['role'] == 'user' %}
<start_of_turn>user
{{ (first_user_extra if loop.first else "") + message['content'] | trim }}<end_of_turn>
{%- elif message['role'] == 'assistant' %}
<start_of_turn>model
{%- if message.get('tool_calls') %}
{%- for tc in message['tool_calls'] %}
<tool_call>
{"name": "{{ tc['function']['name'] }}", "arguments": {{ tc['function']['arguments'] }}}
</tool_call>
{%- endfor %}
{%- else %}
{{ message['content'] | trim }}
{%- endif %}
<end_of_turn>
{%- elif message['role'] == 'tool' %}
<start_of_turn>user
<tool_response>
{{ message['content'] | trim }}
</tool_response><end_of_turn>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<start_of_turn>model
{% endif %}"""

hf_token = getpass("HuggingFace Write Token: ")
api = HfApi(token=hf_token)
headers = {"Authorization": f"Bearer {hf_token}"}

for repo_id in REPOS:
    print(f"\n[+] İşleniyor: {repo_id}")

    # 1. tokenizer_config.json
    url = f"https://huggingface.co/{repo_id}/resolve/main/tokenizer_config.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    config = resp.json()
    config["chat_template"] = NEW_CHAT_TEMPLATE

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="tokenizer_config.json",
        repo_id=repo_id,
        commit_message="Fix: LM Studio tool call desteği için chat_template güncellendi",
        token=hf_token,
    )
    os.unlink(tmp_path)
    print(f"    ✓ tokenizer_config.json")

    # 2. chat_template.jinja
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False, encoding="utf-8") as f:
        f.write(NEW_CHAT_TEMPLATE)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="chat_template.jinja",
        repo_id=repo_id,
        commit_message="Fix: chat_template.jinja güncellendi",
        token=hf_token,
    )
    os.unlink(tmp_path)
    print(f"    ✓ chat_template.jinja")

    print(f"    → https://huggingface.co/{repo_id}")

print("\nSonraki adımlar:")
print("  1. LM Studio'da modeli silin ve yeniden indirin")
print("  2. Model ayarlarında Tool Call Format → Hermes seçin")
