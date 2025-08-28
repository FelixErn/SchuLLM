from datasets import load_dataset
import json
import os

out_file = "data/instructions_de.jsonl"

os.makedirs("data", exist_ok=True)

def save_examples(pairs):
    with open(out_file, "a", encoding="utf-8") as f:
        for instr, resp in pairs:
            if instr and resp:
                f.write(json.dumps(
                    {"instruction": instr.strip(), "response": resp.strip()},
                    ensure_ascii=False
                ) + "\n")

# 1) Alpaca-cleaned-de
print("📥 Lade pinzhenchen/alpaca-cleaned-de ...")
ds1 = load_dataset("pinzhenchen/alpaca-cleaned-de", split="train")
save_examples([(ex["instruction"], ex["output"]) for ex in ds1])

# 2) translated_german_alpaca
print("📥 Lade LEL-A/translated_german_alpaca ...")
ds2 = load_dataset("LEL-A/translated_german_alpaca", split="train")
for ex in ds2:
    try:
        instr = ex.get("inputs", {}).get("_instruction")
        resp = ex.get("inputs", {}).get("output")
        if instr and resp:
            save_examples([(instr, resp)])
    except Exception:
        continue

print("✅ Datensätze gesammelt in:", out_file)