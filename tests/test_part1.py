"""
Checkpoint 1 — Task 1, 2, 3: call_openai, call_openai_mini, compare_models

Chạy:  pytest tests/test_part1.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Đảm bảo import từ template.py ở root repo
sys.path.insert(0, str(Path(__file__).parent.parent))
from template import call_openai, call_openai_mini, compare_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_response(text: str = "Hello") -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# TestCallOpenAI
# ---------------------------------------------------------------------------
class TestCallOpenAI:

    @patch("openai.OpenAI")
    def test_returns_tuple_of_text_and_latency(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Xin chào!")

        result = call_openai("Hello")

        assert isinstance(result, tuple)
        assert len(result) == 2
        text, latency = result
        assert text == "Xin chào!"
        assert isinstance(latency, float)
        assert latency >= 0.0

    @patch("openai.OpenAI")
    def test_passes_parameters_correctly(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response()

        call_openai("Test", temperature=0.3, top_p=0.8, max_tokens=128)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.8
        assert kwargs["max_tokens"] == 128

    @patch("openai.OpenAI")
    def test_message_format_is_user_role(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response()

        call_openai("My prompt")

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "My prompt"

    @patch("openai.OpenAI")
    def test_non_empty_response(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Response text")

        text, _ = call_openai("Hello")
        assert len(text) > 0


# ---------------------------------------------------------------------------
# TestCallOpenAIMini
# ---------------------------------------------------------------------------
class TestCallOpenAIMini:

    @patch("openai.OpenAI")
    def test_returns_tuple_of_text_and_latency(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Mini reply")

        text, latency = call_openai_mini("Hello")

        assert text == "Mini reply"
        assert isinstance(latency, float)
        assert latency >= 0.0

    @patch("openai.OpenAI")
    def test_uses_mini_model(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response()

        call_openai_mini("Test")

        model_used = mock_client.chat.completions.create.call_args.kwargs["model"]
        assert "mini" in model_used.lower()

    @patch("openai.OpenAI")
    def test_forwards_parameters(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response()

        call_openai_mini("Test", temperature=0.1, top_p=0.5, max_tokens=50)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.1
        assert kwargs["top_p"] == 0.5
        assert kwargs["max_tokens"] == 50


# ---------------------------------------------------------------------------
# TestCompareModels
# ---------------------------------------------------------------------------
class TestCompareModels:

    @patch("openai.OpenAI")
    def test_returns_all_five_keys(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Some response here")

        result = compare_models("Test prompt")

        required_keys = {
            "gpt4o_response", "mini_response",
            "gpt4o_latency", "mini_latency",
            "gpt4o_cost_estimate",
        }
        assert required_keys == set(result.keys())

    @patch("openai.OpenAI")
    def test_latency_is_non_negative(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Response")

        result = compare_models("Test")

        assert result["gpt4o_latency"] >= 0
        assert result["mini_latency"] >= 0

    @patch("openai.OpenAI")
    def test_cost_estimate_is_non_negative(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Response")

        result = compare_models("Test")

        assert result["gpt4o_cost_estimate"] >= 0

    @patch("openai.OpenAI")
    def test_responses_are_strings(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Answer")

        result = compare_models("Question")

        assert isinstance(result["gpt4o_response"], str)
        assert isinstance(result["mini_response"], str)
