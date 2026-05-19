"""
Judge registry — tambah judge baru cukup tambah entry di sini.

Keys per entry:
    model        : HF model ID atau nama model API
    provider     : "hf" | "together" | "openai" | "deepseek"
    prompt_mode  : (opsional) default "zeroshot_bing"
"""

JUDGES: dict = {
    # ── Local HF models ──────────────────────────────────────────────────────
    "qwen": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "provider": "hf",
    },
    "sahabat-llama": {
        "model": "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
        "provider": "hf",
    },
    "sahabat-gemma": {
        "model": "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct",
        "provider": "hf",
    },

    # ── OpenAI ───────────────────────────────────────────────────────────────
    # Needs: OPENAI_API_KEY
    "chatgpt": {
        "model": "gpt-4o-mini",
        "provider": "openai",
    },
    "chatgpt-large": {
        "model": "gpt-4o",
        "provider": "openai",
    },

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    # Needs: DEEPSEEK_API_KEY
    "deepseek": {
        "model": "deepseek-chat",
        "provider": "deepseek",
    },
    "deepseek-r1": {
        "model": "deepseek-reasoner",
        "provider": "deepseek",
    },
}


def list_judges() -> None:
    col = 18
    print(f"{'Judge':<{col}} {'Provider':<12} Model")
    print("-" * 60)
    for name, cfg in JUDGES.items():
        print(f"{name:<{col}} {cfg['provider']:<12} {cfg['model']}")
