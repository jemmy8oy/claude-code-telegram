"""Tests for event handlers."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.bus import EventBus
from src.events.handlers import AgentHandler
from src.events.types import AgentResponseEvent, ScheduledEvent, WebhookEvent


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_claude() -> AsyncMock:
    mock = AsyncMock()
    mock.run_command = AsyncMock()
    return mock


@pytest.fixture
def agent_handler(event_bus: EventBus, mock_claude: AsyncMock) -> AgentHandler:
    handler = AgentHandler(
        event_bus=event_bus,
        claude_integration=mock_claude,
        default_working_directory=Path("/tmp/test"),
        default_user_id=42,
    )
    handler.register()
    return handler


class TestAgentHandler:
    """Tests for AgentHandler."""

    async def test_webhook_event_triggers_claude(
        self, event_bus: EventBus, mock_claude: AsyncMock, agent_handler: AgentHandler
    ) -> None:
        """Generic (non-GitHub) webhook events are processed through Claude."""
        mock_response = MagicMock()
        mock_response.content = "Analysis complete"
        mock_claude.run_command.return_value = mock_response

        published: list = []
        original_publish = event_bus.publish

        async def capture_publish(event):  # type: ignore[no-untyped-def]
            published.append(event)
            await original_publish(event)

        event_bus.publish = capture_publish  # type: ignore[assignment]

        # Use a non-GitHub provider so the event reaches the generic webhook
        # path (GitHub events are filtered by label before reaching Claude).
        event = WebhookEvent(
            provider="generic",
            event_type_name="ping",
            payload={"message": "hello"},
            delivery_id="del-1",
        )

        await agent_handler.handle_webhook(event)

        mock_claude.run_command.assert_called_once()
        call_kwargs = mock_claude.run_command.call_args
        assert "generic" in call_kwargs.kwargs["prompt"].lower()

        # Should publish an AgentResponseEvent
        response_events = [e for e in published if isinstance(e, AgentResponseEvent)]
        assert len(response_events) == 1
        assert response_events[0].text == "Analysis complete"

    async def test_scheduled_event_triggers_claude(
        self, event_bus: EventBus, mock_claude: AsyncMock, agent_handler: AgentHandler
    ) -> None:
        """Scheduled events invoke Claude with the job's prompt."""
        mock_response = MagicMock()
        mock_response.content = "Standup summary"
        mock_claude.run_command.return_value = mock_response

        published: list = []
        original_publish = event_bus.publish

        async def capture_publish(event):  # type: ignore[no-untyped-def]
            published.append(event)
            await original_publish(event)

        event_bus.publish = capture_publish  # type: ignore[assignment]

        event = ScheduledEvent(
            job_name="standup",
            prompt="Generate daily standup",
            target_chat_ids=[100],
        )

        await agent_handler.handle_scheduled(event)

        mock_claude.run_command.assert_called_once()
        assert "standup" in mock_claude.run_command.call_args.kwargs["prompt"].lower()

        response_events = [e for e in published if isinstance(e, AgentResponseEvent)]
        assert len(response_events) == 1
        assert response_events[0].chat_id == 100

    async def test_scheduled_event_with_skill(
        self, event_bus: EventBus, mock_claude: AsyncMock, agent_handler: AgentHandler
    ) -> None:
        """Scheduled events with skill_name prepend the skill invocation."""
        mock_response = MagicMock()
        mock_response.content = "Done"
        mock_claude.run_command.return_value = mock_response

        event = ScheduledEvent(
            job_name="standup",
            prompt="morning report",
            skill_name="daily-standup",
            target_chat_ids=[100],
        )

        await agent_handler.handle_scheduled(event)

        prompt = mock_claude.run_command.call_args.kwargs["prompt"]
        assert prompt.startswith("/daily-standup")
        assert "morning report" in prompt

    async def test_claude_error_does_not_propagate(
        self, event_bus: EventBus, mock_claude: AsyncMock, agent_handler: AgentHandler
    ) -> None:
        """Agent errors are logged but don't crash the handler."""
        mock_claude.run_command.side_effect = RuntimeError("SDK error")

        event = WebhookEvent(
            provider="github",
            event_type_name="push",
            payload={},
        )

        # Should not raise
        await agent_handler.handle_webhook(event)

    def test_build_webhook_prompt(self, agent_handler: AgentHandler) -> None:
        """Webhook prompt includes provider and event info."""
        event = WebhookEvent(
            provider="github",
            event_type_name="pull_request",
            payload={"action": "opened", "number": 42},
        )

        prompt = agent_handler._build_webhook_prompt(event)
        assert "github" in prompt.lower()
        assert "pull_request" in prompt
        assert "action: opened" in prompt

    def test_payload_summary_truncation(self, agent_handler: AgentHandler) -> None:
        """Large payloads are truncated in the summary."""
        big_payload = {"key": "x" * 3000}
        summary = agent_handler._summarize_payload(big_payload)
        assert len(summary) <= 2100  # 2000 + truncation message

    # ------------------------------------------------------------------
    # GitHub label-triggered webhook tests
    # ------------------------------------------------------------------

    def _make_github_label_event(
        self,
        label: str = "waiting-for-ai",
        event_type: str = "issues",
        number: int = 42,
        title: str = "Test issue",
    ) -> WebhookEvent:
        """Helper: build a realistic GitHub labeled-issue webhook event."""
        return WebhookEvent(
            provider="github",
            event_type_name=event_type,
            payload={
                "action": "labeled",
                "label": {"name": label},
                "repository": {"full_name": "owner/repo"},
                "issue": {
                    "number": number,
                    "title": title,
                    "html_url": f"https://github.com/owner/repo/issues/{number}",
                    "body": "Please fix this.",
                },
            },
            delivery_id="del-gh-1",
        )

    async def test_github_label_event_sends_pickup_notification_before_claude(
        self, event_bus: EventBus, mock_claude: AsyncMock, agent_handler: AgentHandler
    ) -> None:
        """Pickup notification is published before Claude starts running.

        This is the regression test for issue #28: previously the pickup
        notification was only delivered at the same time as the finished
        notification because handle_webhook() blocked the EventBus loop
        for the entire duration of run_command().

        The fix schedules Claude work via asyncio.create_task() so the
        handler returns (and the EventBus can dispatch the pickup event)
        before run_command() is called.
        """
        pickup_published_before_claude = False
        published: list = []
        original_publish = event_bus.publish

        async def capture_publish(event):  # type: ignore[no-untyped-def]
            published.append(event)
            await original_publish(event)

        event_bus.publish = capture_publish  # type: ignore[assignment]

        # Simulate Claude taking a long time; we record whether the pickup
        # notification was already enqueued when run_command() fires.
        async def slow_claude(**kwargs):  # type: ignore[no-untyped-def]
            nonlocal pickup_published_before_claude
            pickup_events = [
                e for e in published
                if isinstance(e, AgentResponseEvent) and "Picked up" in e.text
            ]
            pickup_published_before_claude = len(pickup_events) > 0
            result = MagicMock()
            result.content = "Done"
            return result

        mock_claude.run_command.side_effect = slow_claude

        event = self._make_github_label_event()

        with (
            patch("src.events.handlers.remove_trigger_label", new_callable=AsyncMock),
            patch("src.events.handlers.AgentHandler._refresh_gh_token", new_callable=AsyncMock),
        ):
            await agent_handler.handle_webhook(event)
            # Allow the background task to run
            await asyncio.sleep(0)
            await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        assert pickup_published_before_claude, (
            "Pickup notification was NOT published before Claude started — "
            "the EventBus is still being blocked by handle_webhook()"
        )

    async def test_github_label_event_publishes_pickup_and_response(
        self, event_bus: EventBus, mock_claude: AsyncMock, agent_handler: AgentHandler
    ) -> None:
        """Both pickup and finished notifications are eventually published."""
        mock_response = MagicMock()
        mock_response.content = "Here is my analysis."
        mock_claude.run_command.return_value = mock_response

        published: list = []
        original_publish = event_bus.publish

        async def capture_publish(event):  # type: ignore[no-untyped-def]
            published.append(event)
            await original_publish(event)

        event_bus.publish = capture_publish  # type: ignore[assignment]

        event = self._make_github_label_event(title="Broken button")

        with (
            patch("src.events.handlers.remove_trigger_label", new_callable=AsyncMock),
            patch("src.events.handlers.AgentHandler._refresh_gh_token", new_callable=AsyncMock),
        ):
            await agent_handler.handle_webhook(event)
            await asyncio.sleep(0)
            await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        response_events = [e for e in published if isinstance(e, AgentResponseEvent)]
        assert len(response_events) == 2, f"Expected 2 AgentResponseEvents, got {len(response_events)}"

        pickup = next(e for e in response_events if "Picked up" in e.text)
        assert 'owner/repo#42' in pickup.text
        assert '"Broken button"' in pickup.text

        finished = next(e for e in response_events if "Picked up" not in e.text)
        assert finished.text == "Here is my analysis."

    async def test_github_label_event_error_posts_error_comment(
        self, event_bus: EventBus, mock_claude: AsyncMock, agent_handler: AgentHandler
    ) -> None:
        """When Claude raises an exception, an error comment is posted to GitHub."""
        mock_claude.run_command.side_effect = RuntimeError("Claude SDK exploded")

        with (
            patch(
                "src.events.handlers.post_error_comment", new_callable=AsyncMock
            ) as mock_error_comment,
            patch("src.events.handlers.AgentHandler._refresh_gh_token", new_callable=AsyncMock),
        ):
            await agent_handler.handle_webhook(self._make_github_label_event())
            await asyncio.sleep(0)
            await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        mock_error_comment.assert_called_once()
        call_kwargs = mock_error_comment.call_args.kwargs
        assert call_kwargs["repo"] == "owner/repo"
        assert call_kwargs["number"] == 42
