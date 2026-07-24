"""
Checkpoint 4 — Part 4: run_assistant (Basic + Scenario)

Chạy:  pytest tests/test_part4.py -v
Chạy chỉ Basic:    pytest tests/test_part4.py -k Basic -v
Chạy chỉ Scenario: pytest tests/test_part4.py -k Scenario -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))
from template import run_assistant


# ---------------------------------------------------------------------------
# Helper — mock streaming chunks
# ---------------------------------------------------------------------------
def _stream_factory(text: str = "AI reply here"):
    """Trả về factory tạo iter chunk mới mỗi lần được gọi."""
    def make_iter(*args, **kwargs):
        chunks = []
        for word in text.split():
            c = MagicMock()
            c.choices[0].delta.content = word + " "
            chunks.append(c)
        # Chunk cuối content = None
        end = MagicMock()
        end.choices[0].delta.content = None
        chunks.append(end)
        return iter(chunks)
    return make_iter


# ---------------------------------------------------------------------------
# TestRunAssistantBasic
# ---------------------------------------------------------------------------
class TestRunAssistantBasic:

    @patch("openai.OpenAI")
    def test_returns_dict(self, MockOpenAI):
        result = run_assistant(get_input=iter(["quit"]).__next__)
        assert isinstance(result, dict)

    @patch("openai.OpenAI")
    def test_returns_correct_keys(self, MockOpenAI):
        result = run_assistant(get_input=iter(["quit"]).__next__)

        required = {"num_turns", "total_tokens", "total_cost", "history"}
        assert required == set(result.keys())

    @patch("openai.OpenAI")
    def test_zero_turns_on_immediate_quit(self, MockOpenAI):
        result = run_assistant(get_input=iter(["quit"]).__next__)
        assert result["num_turns"] == 0
        assert result["total_tokens"] == 0
        assert result["history"] == []

    @patch("openai.OpenAI")
    def test_max_turns_stops_loop(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _stream_factory("Reply")

        call_count = [0]

        def get_inp():
            call_count[0] += 1
            return f"message {call_count[0]}"

        result = run_assistant(max_turns=2, get_input=get_inp)
        assert result["num_turns"] == 2


# ---------------------------------------------------------------------------
# TestRunAssistantScenario
# ---------------------------------------------------------------------------
class TestRunAssistantScenario:

    @patch("openai.OpenAI")
    def test_persona_as_system_message(self, MockOpenAI):
        """Persona phải xuất hiện là system message đầu tiên trong mỗi lần gọi API."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _stream_factory("Hi there")

        persona = "You are a pirate assistant."
        inputs = iter(["Hello", "quit"])
        run_assistant(persona=persona, get_input=lambda: next(inputs))

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == persona

    @patch("openai.OpenAI")
    def test_history_has_user_and_assistant(self, MockOpenAI):
        """Sau 1 lượt: history phải có đúng 2 message (user + assistant)."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _stream_factory("AI reply")

        inputs = iter(["Hello", "quit"])
        result = run_assistant(max_turns=1, get_input=lambda: next(inputs))

        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    @patch("openai.OpenAI")
    def test_history_limited_to_six_messages(self, MockOpenAI):
        """History không được vượt quá 6 message (3 lượt) dù chạy nhiều lượt hơn."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _stream_factory("Reply")

        call_count = [0]

        def get_inp():
            call_count[0] += 1
            return f"msg {call_count[0]}"

        result = run_assistant(max_turns=5, get_input=get_inp)
        assert len(result["history"]) <= 6

    @patch("openai.OpenAI")
    def test_stats_accumulate_correctly(self, MockOpenAI):
        """num_turns, total_tokens và total_cost phải tăng theo số lượt thật."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _stream_factory(
            "This is a test response"
        )

        call_count = [0]

        def get_inp():
            call_count[0] += 1
            return "question"

        result = run_assistant(max_turns=2, get_input=get_inp)

        assert result["num_turns"] == 2
        assert result["total_tokens"] > 0
        assert result["total_cost"] >= 0

    @patch("openai.OpenAI")
    def test_stream_true_is_used(self, MockOpenAI):
        """API phải được gọi với stream=True."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _stream_factory("Hi")

        inputs = iter(["Hello", "quit"])
        run_assistant(max_turns=1, get_input=lambda: next(inputs))

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("stream") is True
