"""
Checkpoint 3 — Task 3.1, 3.2: streaming_chatbot, retry_with_backoff

Chạy:  pytest tests/test_part3.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from template import streaming_chatbot, retry_with_backoff


# ---------------------------------------------------------------------------
# TestStreamingChatbot
# ---------------------------------------------------------------------------
class TestStreamingChatbot:

    @patch("builtins.input", side_effect=["quit"])
    @patch("openai.OpenAI")
    def test_exits_cleanly_on_quit(self, MockOpenAI, mock_input):
        """Chatbot phải thoát sạch khi nhận 'quit', không treo."""
        try:
            streaming_chatbot()
        except SystemExit:
            pass
        # Nếu đến được đây mà không treo → pass

    @patch("builtins.input", side_effect=["exit"])
    @patch("openai.OpenAI")
    def test_exits_cleanly_on_exit(self, MockOpenAI, mock_input):
        """Chatbot phải thoát sạch khi nhận 'exit'."""
        try:
            streaming_chatbot()
        except SystemExit:
            pass

    @patch("builtins.input", side_effect=["QUIT"])
    @patch("openai.OpenAI")
    def test_case_insensitive_exit(self, MockOpenAI, mock_input):
        """'QUIT' (hoa) cũng phải thoát."""
        try:
            streaming_chatbot()
        except SystemExit:
            pass


# ---------------------------------------------------------------------------
# TestRetryWithBackoff
# ---------------------------------------------------------------------------
class TestRetryWithBackoff:

    def test_succeeds_immediately(self):
        result = retry_with_backoff(lambda: "success")
        assert result == "success"

    def test_returns_correct_value(self):
        result = retry_with_backoff(lambda: 42)
        assert result == 42

    def test_succeeds_after_retries(self):
        state = {"attempts": 0}

        def flaky():
            state["attempts"] += 1
            if state["attempts"] < 3:
                raise ValueError("not yet")
            return "ok"

        result = retry_with_backoff(flaky, max_retries=3, base_delay=0.001)
        assert result == "ok"
        assert state["attempts"] == 3

    def test_raises_when_all_retries_exhausted(self):
        import pytest

        def always_fail():
            raise RuntimeError("permanent error")

        with pytest.raises(RuntimeError, match="permanent error"):
            retry_with_backoff(always_fail, max_retries=2, base_delay=0.001)

    def test_total_calls_equals_one_plus_max_retries(self):
        state = {"calls": 0}

        def counter():
            state["calls"] += 1
            raise ValueError("fail")

        try:
            retry_with_backoff(counter, max_retries=2, base_delay=0.001)
        except ValueError:
            pass

        # 1 lần gọi đầu + 2 lần retry = 3
        assert state["calls"] == 3

    def test_re_raises_same_exception_type(self):
        import pytest

        def raise_value_error():
            raise ValueError("specific message")

        with pytest.raises(ValueError, match="specific message"):
            retry_with_backoff(raise_value_error, max_retries=1, base_delay=0.001)
