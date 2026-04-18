# ── model.py ──────────────────────────────────────────────────────────────────
# Mistral-7B loading and inference

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLM_MODEL_NAME

tokenizer  = None
llm_model  = None


def load_model(hf_token: str):
    """Load Mistral-7B in 4-bit NF4 quantization onto GPU."""
    global tokenizer, llm_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("Loading Mistral-7B (~3-4 mins)...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, token=hf_token)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )

    tokenizer.pad_token = tokenizer.eos_token
    llm_model.config.pad_token_id = llm_model.config.eos_token_id

    print("Mistral-7B loaded!")


def call_mistral(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Run a single inference call on Mistral-7B."""
    prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to("cuda")

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
