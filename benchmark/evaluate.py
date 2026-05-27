#!/usr/bin/env python3
"""
Turkish Agentic LLM Benchmark — Tool Calling Accuracy Evaluator

Hem base modeli hem de fine-tuned modeli 100 Türkçe tool-call sorusuyla
değerlendirir; format geçerliliği, fonksiyon adı doğruluğu ve tam eşleşme
metriklerini hesaplar.

Kullanım:
  # Tek model değerlendirme:
  python evaluate.py --model AlicanKiraz0/Kizagan-E4B-Turkish-Reasoning-Model --output results/base.json
  python evaluate.py --model Tuguberk/Kizagan-E4B-FunctionCalling-TR --output results/finetuned.json

  # İki sonucu karşılaştırma:
  python evaluate.py --compare results/base.json results/finetuned.json

  # Hızlı test (ilk 10 soru):
  python evaluate.py --model Tuguberk/Kizagan-E4B-FunctionCalling-TR --limit 10 --output results/test.json

Gereksinimler:
  pip install transformers torch accelerate tqdm
  # CUDA için ek:  pip install bitsandbytes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sistem istemi şablonu — eğitim verisindeki Hermes formatıyla birebir aynı
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = (
    "Fonksiyon çağırma yeteneğine sahip bir yapay zeka modelisiniz. "
    "Size <tools> </tools> XML etiketleri içinde fonksiyon imzaları sağlanmıştır. "
    "Kullanıcı sorgusuna yardımcı olmak için bir veya daha fazla fonksiyonu çağırabilirsiniz. "
    "Fonksiyonlara hangi değerlerin girileceği konusunda varsayımlarda bulunmayın.\n"
    "<tools>\n{tools}\n</tools>\n"
    "Her fonksiyon çağrısı için, aşağıdaki şema ile <tool_call> </tool_call> etiketleri "
    "içinde fonksiyon adı ve argümanları içeren bir json nesnesi döndürün:\n"
    "<tool_call>\n"
    '{{"name": <fonksiyon-adı>, "arguments": <args-sözlüğü>}}\n'
    "</tool_call>"
)


# ===========================================================================
# Yardımcı: string normalleştirme ve değer karşılaştırma
# ===========================================================================

def _normalize(s: str) -> str:
    """Karşılaştırma için string'i normalize eder: Türkçe karakter → ASCII, küçük harf, trim."""
    s = str(s).strip()
    # Python'da "İ".lower() == "i̇" (combining dot) verir; önce ASCII'ye çevirmeliyiz.
    for tr, en in [
        ("İ", "I"), ("ı", "i"), ("Ş", "S"), ("ş", "s"),
        ("Ğ", "G"), ("ğ", "g"), ("Ç", "C"), ("ç", "c"),
        ("Ö", "O"), ("ö", "o"), ("Ü", "U"), ("ü", "u"),
    ]:
        s = s.replace(tr, en)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _values_match(predicted: Any, expected: Any) -> bool:
    """İki değerin semantik olarak eşit olup olmadığını kontrol eder."""
    # Sayısal karşılaştırma
    try:
        return float(str(predicted)) == float(str(expected))
    except (ValueError, TypeError):
        pass
    # Normalize edilmiş string karşılaştırması
    return _normalize(str(predicted)) == _normalize(str(expected))


# ===========================================================================
# Skorlama
# ===========================================================================

def score_prediction(
    predicted_calls: list[dict],
    expected: dict | list[dict],
) -> dict:
    """
    Tahmin edilen tool call'ları beklenenlerle karşılaştırır.

    Döndürülen metrikler:
      format_valid  — Çıktıda parse edilebilir en az 1 tool_call var mı?
      name_match    — Tüm beklenen fonksiyon adları tahminlerde bulundu mu?
      exact_match   — Adlar VE tüm argümanlar doğru mu?
    """
    if not predicted_calls:
        return {"format_valid": 0, "name_match": 0, "exact_match": 0}

    expected_list: list[dict] = expected if isinstance(expected, list) else [expected]
    n = len(expected_list)
    name_hits = 0
    exact_hits = 0

    for exp in expected_list:
        exp_name = exp["name"]
        exp_args = exp.get("arguments", {})

        # Aynı fonksiyon adına sahip ilk tahmini bul
        best = next((p for p in predicted_calls if p.get("name") == exp_name), None)
        if best is None:
            continue

        name_hits += 1

        # Tüm beklenen argümanlar eşleşiyor mu?
        pred_args = best.get("arguments", {})
        if all(
            key in pred_args and _values_match(pred_args[key], val)
            for key, val in exp_args.items()
        ):
            exact_hits += 1

    return {
        "format_valid": 1,
        "name_match": int(name_hits >= n),
        "exact_match": int(exact_hits >= n),
        # Kısmi eşleşme oranları (raporlama için)
        "name_match_ratio": name_hits / n,
        "exact_match_ratio": exact_hits / n,
    }


# ===========================================================================
# Tool call ayrıştırma
# ===========================================================================

def parse_tool_calls(text: str) -> list[dict]:
    """
    Model çıktısından <tool_call>…</tool_call> bloklarını ayıklar.
    Birden fazla çağrı varsa hepsini döndürür.
    """
    calls: list[dict] = []

    # Birincil yöntem: <tool_call>JSON</tool_call>
    for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        raw = raw.strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # JSON içindeki ilk nesneyi bulmayı dene
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group())
                except json.JSONDecodeError:
                    continue
            else:
                continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)

    # Geri dönüş: "name" ve "arguments" içeren herhangi bir JSON nesnesi
    if not calls:
        for m in re.finditer(r'\{[^{}]*"name"\s*:[^{}]*\}', text):
            try:
                obj = json.loads(m.group())
                if "name" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

    return calls


# ===========================================================================
# Model yükleme & çıkarım
# ===========================================================================

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[+] Tokenizer yükleniyor: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}

    if device == "cuda":
        # CUDA'da 4-bit quantization ile bellek tasarrufu
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
            print("[+] CUDA: 4-bit QLoRA quantization aktif.")
        except ImportError:
            load_kwargs["device_map"] = "auto"
            print("[!] bitsandbytes bulunamadı; tam hassasiyetle yükleniyor.")
    elif device == "mps":
        # Apple Silicon: MPS backend
        print("[+] Apple Silicon MPS kullanılıyor.")
    else:
        print("[+] CPU modunda çalışılıyor (yavaş olabilir).")

    print(f"[+] Model yükleniyor: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if device == "mps":
        model = model.to("mps")
    elif device == "cpu":
        model = model.to("cpu")

    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"[+] Model hazır. (CUDA bellek: {vram:.1f} GB)")
    return model, tokenizer


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    device: str,
    max_new_tokens: int = 512,
) -> str:
    """Greedy decoding ile deterministik yanıt üretir."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    target = device if device != "cpu" else "cpu"
    input_ids = input_ids.to(target)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy → tam tekrarlanabilirlik
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ===========================================================================
# Değerlendirme döngüsü
# ===========================================================================

def build_messages(question: dict) -> list[dict]:
    """Sorudan Gemma formatında chat mesajları oluşturur."""
    tools_json = json.dumps(question["tools"], ensure_ascii=False, indent=2)
    system_block = SYSTEM_PROMPT_TEMPLATE.format(tools=tools_json)
    return [{"role": "user", "content": f"{system_block}\n\n{question['query']}"}]


def run_evaluation(
    model,
    tokenizer,
    questions: list[dict],
    device: str,
) -> list[dict]:
    results = []

    for q in tqdm(questions, desc="Değerlendiriliyor", unit="soru"):
        messages = build_messages(q)
        output = ""
        elapsed = 0.0

        try:
            t0 = time.perf_counter()
            output = generate_response(model, tokenizer, messages, device)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            tqdm.write(f"\n[HATA] Soru {q['id']}: {exc}")

        predicted_calls = parse_tool_calls(output)
        scores = score_prediction(predicted_calls, q["expected"])

        results.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "query": q["query"],
            "expected": q["expected"],
            "output": output,
            "predicted_calls": predicted_calls,
            "scores": scores,
            "elapsed_sec": round(elapsed, 3),
        })

    return results


# ===========================================================================
# Raporlama
# ===========================================================================

def print_summary(results: list[dict], model_label: str = "Model") -> None:
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r)

    total = len(results)
    fmt = sum(r["scores"]["format_valid"] for r in results)
    name = sum(r["scores"]["name_match"] for r in results)
    exact = sum(r["scores"]["exact_match"] for r in results)

    header = f"  SONUÇLAR: {model_label}  "
    print(f"\n{'='*65}")
    print(header.center(65))
    print(f"{'='*65}")
    print(f"{'Kategori':<22} {'N':>4}  {'Format%':>8}  {'Ad%':>8}  {'Tam%':>8}")
    print(f"{'-'*65}")

    for cat, cat_rs in sorted(cats.items()):
        n = len(cat_rs)
        cf = sum(r["scores"]["format_valid"] for r in cat_rs)
        cn = sum(r["scores"]["name_match"] for r in cat_rs)
        ce = sum(r["scores"]["exact_match"] for r in cat_rs)
        print(
            f"{cat:<22} {n:>4}  {100*cf/n:>7.1f}%  "
            f"{100*cn/n:>7.1f}%  {100*ce/n:>7.1f}%"
        )

    print(f"{'-'*65}")
    print(
        f"{'TOPLAM':<22} {total:>4}  {100*fmt/total:>7.1f}%  "
        f"{100*name/total:>7.1f}%  {100*exact/total:>7.1f}%"
    )
    print(f"{'='*65}")


def compare_results(base_path: str, ft_path: str) -> None:
    """İki sonuç dosyasını karşılaştırır ve özet tablo yazar."""
    with open(base_path, encoding="utf-8") as f:
        base_data = json.load(f)
    with open(ft_path, encoding="utf-8") as f:
        ft_data = json.load(f)

    base_rs = base_data["results"]
    ft_rs = ft_data["results"]
    base_label = base_data.get("model_id", "Base Model")
    ft_label = ft_data.get("model_id", "Fine-tuned Model")

    print_summary(base_rs, base_label)
    print_summary(ft_rs, ft_label)

    print(f"\n{'='*70}")
    print("  KARŞILAŞTIRMA: Base vs Fine-tuned".center(70))
    print(f"{'='*70}")
    print(f"{'Metrik':<22} {'Base':>14} {'Fine-tuned':>14} {'Delta':>12}")
    print(f"{'-'*70}")

    metrics = [
        ("Format Geçerliliği %", "format_valid"),
        ("Fonksiyon Adı %",      "name_match"),
        ("Tam Eşleşme %",        "exact_match"),
    ]

    for label, key in metrics:
        b = sum(r["scores"][key] for r in base_rs) / len(base_rs) * 100
        f = sum(r["scores"][key] for r in ft_rs) / len(ft_rs) * 100
        d = f - b
        sign = "+" if d >= 0 else ""
        print(f"{label:<22} {b:>13.1f}% {f:>13.1f}% {sign}{d:>9.1f}%")

    print(f"{'='*70}")

    # Fine-tuned'ın geliştiği örnekler
    improvements = [
        (b, f)
        for b, f in zip(base_rs, ft_rs)
        if f["scores"]["exact_match"] > b["scores"]["exact_match"]
    ]
    regressions = [
        (b, f)
        for b, f in zip(base_rs, ft_rs)
        if f["scores"]["exact_match"] < b["scores"]["exact_match"]
    ]

    if improvements:
        print(f"\n✅  Fine-tuned'ın iyileştirdiği {len(improvements)} soru (ilk 5):")
        for bq, fq in improvements[:5]:
            print(f"  [{bq['id']:>3}] {bq['query'][:55]}")
            print(f"         Beklenen  : {bq['expected']}")
            print(f"         Base pred : {bq['predicted_calls'][:1]}")
            print(f"         FT pred   : {fq['predicted_calls'][:1]}")

    if regressions:
        print(f"\n⚠️   Fine-tuned'ın gerilediği {len(regressions)} soru (ilk 5):")
        for bq, fq in regressions[:5]:
            print(f"  [{bq['id']:>3}] {bq['query'][:55]}")

    # Zorluk bazında karşılaştırma
    difficulties = sorted({r["difficulty"] for r in base_rs})
    print(f"\n{'Zorluk':<12} {'Base Tam%':>12} {'FT Tam%':>12} {'Delta':>10}")
    print(f"{'-'*48}")
    for diff in difficulties:
        b_sub = [r for r in base_rs if r["difficulty"] == diff]
        f_sub = [r for r in ft_rs  if r["difficulty"] == diff]
        b_e = sum(r["scores"]["exact_match"] for r in b_sub) / max(len(b_sub), 1) * 100
        f_e = sum(r["scores"]["exact_match"] for r in f_sub) / max(len(f_sub), 1) * 100
        d = f_e - b_e
        sign = "+" if d >= 0 else ""
        print(f"{diff:<12} {b_e:>11.1f}% {f_e:>11.1f}% {sign}{d:>8.1f}%")


# ===========================================================================
# CLI giriş noktası
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Türkçe LLM Tool-Calling Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, help="HuggingFace model ID veya yerel yol")
    parser.add_argument(
        "--questions", type=str, default="questions.json",
        help="Soru dosyasının yolu (varsayılan: questions.json)"
    )
    parser.add_argument("--output", type=str, help="Sonuçların kaydedileceği JSON dosyası")
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASE_JSON", "FT_JSON"),
        help="İki sonuç dosyasını karşılaştır"
    )
    parser.add_argument(
        "--device", choices=["cuda", "mps", "cpu", "auto"], default="auto",
        help="Hesaplama cihazı (varsayılan: auto)"
    )
    parser.add_argument("--limit", type=int, help="Test edilecek maksimum soru sayısı")
    parser.add_argument(
        "--category", type=str,
        help="Sadece bu kategorideki soruları değerlendir"
    )
    args = parser.parse_args()

    # Karşılaştırma modu
    if args.compare:
        compare_results(*args.compare)
        return

    if not args.model:
        parser.error("--model gereklidir. Yardım için: python evaluate.py --help")

    # Sorular
    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"[HATA] Soru dosyası bulunamadı: {questions_path}")

    with questions_path.open(encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        print(f"[+] Filtre: yalnızca '{args.category}' kategorisi — {len(questions)} soru")

    if args.limit:
        questions = questions[: args.limit]

    print(f"[+] Toplam soru: {len(questions)}")

    # Çıktı yolu
    output_path = args.output
    if not output_path:
        safe_name = args.model.replace("/", "_").replace("\\", "_")
        output_path = f"results/{safe_name}.json"

    # Cihaz
    device = detect_device() if args.device == "auto" else args.device
    print(f"[+] Cihaz: {device}")

    # Model yükle
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Değerlendirme
    t_start = time.perf_counter()
    results = run_evaluation(model, tokenizer, questions, device)
    total_time = time.perf_counter() - t_start

    # Özet
    print_summary(results, args.model)
    print(f"\n[+] Toplam süre: {total_time:.1f}s | Soru başına: {total_time/len(questions):.1f}s")

    # Kaydet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": args.model,
                "device": device,
                "num_questions": len(questions),
                "total_time_sec": round(total_time, 2),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[+] Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
