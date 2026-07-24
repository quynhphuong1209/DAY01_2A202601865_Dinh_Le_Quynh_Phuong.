"""
Day 1 — LLM API Foundation
K3 Ngày 1: Khám Phá LLM API

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Pricing per 1,000 tokens (USD) — dùng để ước tính chi phí
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Model constants — đọc từ .env nếu có (hỗ trợ NVIDIA NIM)
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Task 1 — Gọi GPT-4o và đo latency
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API và trả về (response_text, latency_seconds).

    Args:
        prompt:      Câu hỏi / yêu cầu gửi đến model.
        model:       Model sẽ dùng (mặc định: gpt-4o).
        temperature: Mức sáng tạo (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Giới hạn độ dài output.

    Returns:
        tuple(response_text: str, latency_seconds: float)
    """
    from openai import OpenAI  # import TRONG hàm — bắt buộc để mock hoạt động

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    text = response.choices[0].message.content
    return text, latency


# ---------------------------------------------------------------------------
# Task 2 — Wrapper gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Tái sử dụng call_openai với OPENAI_MINI_MODEL.
    Trả về (response_text, latency_seconds).
    """
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — So sánh GPT-4o với GPT-4o-mini trên cùng prompt
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả GPT-4o và GPT-4o-mini với cùng prompt.

    Returns:
        dict với đúng 5 key:
            - gpt4o_response (str)
            - mini_response (str)
            - gpt4o_latency (float)
            - mini_latency (float)
            - gpt4o_cost_estimate (float): ước tính USD chỉ cho output GPT-4o
    """
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    # Ước tính thô: 0.75 từ ≈ 1 token
    gpt4o_cost_estimate = (
        (len(gpt4o_response.split()) / 0.75) / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }


# ---------------------------------------------------------------------------
# Task 2.1 — Thêm system prompt vào hội thoại
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gửi system_prompt (định persona) kèm user_prompt đến API.

    Returns:
        tuple(response_text: str, latency_seconds: float)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    return response.choices[0].message.content, latency


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token với fallback
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token trong text bằng tiktoken.
    Nếu model lạ hoặc không có mạng, fallback: max(1, len(text) // 4).

    Returns:
        int — số token (tối thiểu 1)
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Tính chi phí input + output + total
# ---------------------------------------------------------------------------
def estimate_cost(
    prompt: str,
    response: str,
    model: str = OPENAI_MODEL,
) -> dict:
    """
    Ước tính USD cho một lần gọi API.

    Returns:
        dict với đúng 5 key:
            input_tokens, output_tokens, input_cost, output_cost, total_cost
    """
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])

    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)

    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot CLI dùng OpenAI streaming.
    Giữ tối đa 3 lượt gần nhất (6 message).
    Gõ 'quit' hoặc 'exit' để thoát.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict] = []

    while True:
        try:
            user_msg = input("Bạn: ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_msg.strip().lower() in ("quit", "exit"):
            break

        history.append({"role": "user", "content": user_msg})

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True,
        )

        print("AI: ", end="", flush=True)
        reply_parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        reply = "".join(reply_parts)
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu lỗi, thử lại tối đa max_retries lần với delay tăng gấp đôi.
    Ném lại exception cuối nếu hết lượt.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Part 4 — Trợ lý CLI hoàn chỉnh (ghép tất cả)
# ---------------------------------------------------------------------------
def run_assistant(
    persona: str = "You are a helpful assistant.",
    max_turns: int | None = None,
    get_input: Callable[[], str] | None = None,
) -> dict:
    """
    Chạy trợ lý CLI đa lượt: persona + streaming + history + retry + thống kê.

    Args:
        persona:    System prompt định persona cho assistant.
        max_turns:  Số lượt tối đa (None = không giới hạn).
        get_input:  Hàm đọc input người dùng (mặc định: input()).

    Returns:
        dict với 4 key: num_turns, total_tokens, total_cost, history
    """
    if get_input is None:
        get_input = input

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    history: list[dict] = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        # Kiểm tra max_turns TRƯỚC khi gọi get_input
        if max_turns is not None and num_turns >= max_turns:
            break

        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )

        # Gọi API qua retry để chịu lỗi tạm thời
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )

        reply_parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        reply = "".join(reply_parts)

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]

        # Cộng dồn thống kê
        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Bonus A — So sánh nhiều prompt một lúc
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho mỗi prompt trong danh sách.
    Mỗi kết quả có thêm key 'prompt' chứa câu hỏi gốc.
    """
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({**comparison, "prompt": prompt})
    return results


# ---------------------------------------------------------------------------
# Bonus B — Xuất bảng Markdown dễ đọc
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Chuyển kết quả batch_compare thành bảng Markdown.
    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    """
    def shorten(text: str, max_len: int = 40) -> str:
        return text[:max_len] + "..." if len(text) > max_len else text

    header = "| Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency |"
    separator = "|---|---|---|---|---|"

    rows = [header, separator]
    for r in results:
        row = (
            f"| {shorten(r['prompt'])} "
            f"| {shorten(r['gpt4o_response'])} "
            f"| {shorten(r['mini_response'])} "
            f"| {r['gpt4o_latency']:.2f}s "
            f"| {r['mini_latency']:.2f}s |"
        )
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_assistant()
