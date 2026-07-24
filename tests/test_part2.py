"""
Checkpoint 2 — Task 2.1, 2.2, 2.3: chat_with_system_prompt, count_tokens, estimate_cost

Chạy:  pytest tests/test_part2.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from template import chat_with_system_prompt, count_tokens, estimate_cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_response(text: str = "Reply") -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# TestSystemPrompt
# ---------------------------------------------------------------------------
class TestSystemPrompt:

    @patch("openai.OpenAI")
    def test_returns_text_and_latency(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Answer")

        text, latency = chat_with_system_prompt("You are a poet.", "Write a haiku.")

        assert text == "Answer"
        assert isinstance(latency, float)
        assert latency >= 0.0

    @patch("openai.OpenAI")
    def test_system_message_is_first(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response()

        chat_with_system_prompt("You are helpful.", "Hello")

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    @patch("openai.OpenAI")
    def test_user_message_is_second(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response()

        chat_with_system_prompt("System instruction", "User question")

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User question"

    @patch("openai.OpenAI")
    def test_system_prompt_affects_model(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Different")

        text, _ = chat_with_system_prompt(
            "Always reply in UPPERCASE.", "hello"
        )
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# TestCountTokens
# ---------------------------------------------------------------------------
class TestCountTokens:

    def test_returns_positive_integer(self):
        result = count_tokens("Hello world")
        assert isinstance(result, int)
        assert result >= 1

    def test_unknown_model_fallback(self):
        # Model không tồn tại → fallback, không được crash
        result = count_tokens("Hello world", model="nonexistent-model-xyz-999")
        assert isinstance(result, int)
        assert result >= 1

    def test_empty_string_returns_at_least_one(self):
        result = count_tokens("", model="nonexistent-model-xyz-999")
        assert result >= 1

    def test_longer_text_has_more_tokens(self):
        short_tokens = count_tokens("Hi")
        long_tokens = count_tokens(
            "This is a much longer sentence with many words that should produce more tokens."
        )
        assert long_tokens > short_tokens

    def test_known_model_returns_reasonable_count(self):
        # "Hello" trong tiếng Anh ≈ 1 token
        result = count_tokens("Hello")
        assert 1 <= result <= 5  # token count reasonable


# ---------------------------------------------------------------------------
# TestEstimateCost
# ---------------------------------------------------------------------------
class TestEstimateCost:

    def test_returns_all_five_keys(self):
        result = estimate_cost("Hello prompt", "Hello response")

        required_keys = {
            "input_tokens", "output_tokens",
            "input_cost", "output_cost", "total_cost",
        }
        assert required_keys == set(result.keys())

    def test_total_cost_equals_sum_of_parts(self):
        result = estimate_cost("Test prompt", "Test response")

        expected = result["input_cost"] + result["output_cost"]
        assert abs(result["total_cost"] - expected) < 1e-12

    def test_costs_are_non_negative(self):
        result = estimate_cost("Test", "Response")

        assert result["input_cost"] >= 0
        assert result["output_cost"] >= 0
        assert result["total_cost"] >= 0

    def test_token_counts_are_positive(self):
        result = estimate_cost("Hello", "World")

        assert result["input_tokens"] >= 1
        assert result["output_tokens"] >= 1

    def test_unknown_model_uses_fallback_pricing(self):
        # Model lạ → không crash, dùng giá mặc định
        result = estimate_cost("Test", "Response", model="gpt-99-nonexistent")
        assert "total_cost" in result
        assert result["total_cost"] >= 0

    def test_longer_text_costs_more(self):
        short = estimate_cost("Hi", "OK")
        long = estimate_cost(
            "This is a much longer prompt with many words.",
            "This is a much longer response with many words and sentences.",
        )
        assert long["total_cost"] >= short["total_cost"]
