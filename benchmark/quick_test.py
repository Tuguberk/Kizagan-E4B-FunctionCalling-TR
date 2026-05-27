#!/usr/bin/env python3
"""
Hızlı sanity-check: model yüklenmeden scoring ve parsing mantığını test eder.
Modeli indirmeden önce çalıştırın.

Kullanım:
  python quick_test.py
"""

import json
import sys
from pathlib import Path

# evaluate.py'den gerekli fonksiyonları import et
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import parse_tool_calls, score_prediction, _normalize

# ─── Test 1: Soru dosyası formatı ────────────────────────────────────────────
print("=== Test 1: questions.json formatı ===")
with open("questions.json", encoding="utf-8") as f:
    data = json.load(f)

questions = data["questions"]
assert len(questions) == 100, f"100 soru beklendi, {len(questions)} var"
for q in questions:
    assert "id" in q and "category" in q and "difficulty" in q
    assert "tools" in q and "query" in q and "expected" in q
    assert isinstance(q["expected"], dict) and "name" in q["expected"]
print(f"✓ {len(questions)} soru, tüm alanlar geçerli")

# Kategori dağılımı
from collections import Counter
cat_counts = Counter(q["category"] for q in questions)
diff_counts = Counter(q["difficulty"] for q in questions)
print(f"\nKategori dağılımı:")
for cat, n in sorted(cat_counts.items()):
    print(f"  {cat:<22}: {n:>3} soru")
print(f"\nZorluk dağılımı:")
for diff, n in sorted(diff_counts.items()):
    print(f"  {diff:<10}: {n:>3} soru")

# ─── Test 2: Tool call ayrıştırma ────────────────────────────────────────────
print("\n=== Test 2: parse_tool_calls ===")

test_cases = [
    # Standart Hermes formatı
    (
        '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "İstanbul"}}\n</tool_call>',
        [{"name": "get_current_weather", "arguments": {"city": "İstanbul"}}],
    ),
    # Çift tool call
    (
        '<tool_call>\n{"name": "search_web", "arguments": {"query": "python"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "search_news", "arguments": {"query": "ai"}}\n</tool_call>',
        [
            {"name": "search_web", "arguments": {"query": "python"}},
            {"name": "search_news", "arguments": {"query": "ai"}},
        ],
    ),
    # Boş çıktı
    ("Merhaba, size nasıl yardımcı olabilirim?", []),
]

for i, (text, expected) in enumerate(test_cases, 1):
    result = parse_tool_calls(text)
    assert result == expected, f"Test {i} başarısız:\n  Beklenen: {expected}\n  Alınan:   {result}"
    print(f"  ✓ Test {i}: {'%d tool call' % len(expected) if expected else 'boş çıktı'} doğru ayrıştırıldı")

# ─── Test 3: Skorlama ─────────────────────────────────────────────────────────
print("\n=== Test 3: score_prediction ===")

score_tests = [
    # Tam eşleşme
    (
        [{"name": "get_current_weather", "arguments": {"city": "İstanbul"}}],
        {"name": "get_current_weather", "arguments": {"city": "İstanbul"}},
        {"format_valid": 1, "name_match": 1, "exact_match": 1},
    ),
    # Türkçe karakter normalizasyonu
    (
        [{"name": "get_current_weather", "arguments": {"city": "istanbul"}}],
        {"name": "get_current_weather", "arguments": {"city": "İstanbul"}},
        {"format_valid": 1, "name_match": 1, "exact_match": 1},
    ),
    # Yanlış fonksiyon adı
    (
        [{"name": "get_weather", "arguments": {"city": "İstanbul"}}],
        {"name": "get_current_weather", "arguments": {"city": "İstanbul"}},
        {"format_valid": 1, "name_match": 0, "exact_match": 0},
    ),
    # Doğru ad, eksik argüman
    (
        [{"name": "get_weather_forecast", "arguments": {"city": "Ankara"}}],
        {"name": "get_weather_forecast", "arguments": {"city": "Ankara", "days": 5}},
        {"format_valid": 1, "name_match": 1, "exact_match": 0},
    ),
    # Sayısal eşleşme (int vs string)
    (
        [{"name": "convert_currency", "arguments": {"amount": "100", "from_currency": "USD", "to_currency": "TRY"}}],
        {"name": "convert_currency", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "TRY"}},
        {"format_valid": 1, "name_match": 1, "exact_match": 1},
    ),
    # Hiç tahmin yok
    (
        [],
        {"name": "get_current_weather", "arguments": {"city": "İstanbul"}},
        {"format_valid": 0, "name_match": 0, "exact_match": 0},
    ),
]

for i, (predicted, expected, expected_scores) in enumerate(score_tests, 1):
    scores = score_prediction(predicted, expected)
    for key in ["format_valid", "name_match", "exact_match"]:
        assert scores[key] == expected_scores[key], (
            f"Test {i} [{key}] başarısız: beklenen={expected_scores[key]}, alınan={scores[key]}\n"
            f"  Tahmin: {predicted}\n  Beklenen: {expected}"
        )
    print(f"  ✓ Test {i}: format={scores['format_valid']} ad={scores['name_match']} tam={scores['exact_match']}")

# ─── Test 4: String normalleştirme ────────────────────────────────────────────
print("\n=== Test 4: Türkçe karakter normalizasyonu ===")
pairs = [
    ("İstanbul", "istanbul"), ("Şişli", "sisli"),
    ("Üsküdar", "uskudar"), ("Çankaya", "cankaya"),
]
for orig, exp in pairs:
    assert _normalize(orig) == exp, f"_normalize({orig!r}) = {_normalize(orig)!r}, beklenen {exp!r}"
print("  ✓ Tüm Türkçe karakterler doğru normalleştirildi")

# ─── Özet ────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("✅  Tüm testler geçti! Şimdi evaluate.py'yi çalıştırabilirsiniz.")
print("="*50)
print("\nKomutlar:")
print("  # Base model:")
print("  python evaluate.py \\")
print("    --model AlicanKiraz0/Kizagan-E4B-Turkish-Reasoning-Model \\")
print("    --output results/base.json")
print()
print("  # Fine-tuned model:")
print("  python evaluate.py \\")
print("    --model Tuguberk/Kizagan-E4B-FunctionCalling-TR \\")
print("    --output results/finetuned.json")
print()
print("  # Karşılaştırma:")
print("  python evaluate.py --compare results/base.json results/finetuned.json")
