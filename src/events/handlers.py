"""Event handlers that bridge the event bus to Claude and Telegram.

AgentHandler: translates events into ClaudeIntegration.run_command() calls.
NotificationHandler: subscribes to AgentResponseEvent and delivers to Telegram.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..claude.facade import ClaudeIntegration
from ..github.label_workflow import (
    LABEL_ACTION_READY,
    LABEL_WAITING_FOR_AI,
    post_comment_and_remove_label,
    post_error_comment,
)
from .bus import Event, EventBus
from .types import AgentResponseEvent, ScheduledEvent, WebhookEvent

# Labels that trigger Claude on issues/PRs
_TRIGGER_LABELS = {LABEL_WAITING_FOR_AI, LABEL_ACTION_READY}

logger = structlog.get_logger()


class AgentHandler:
    """Translates incoming events into Claude agent executions.

    Webhook and scheduled events are converted into prompts and sent
    to ClaudeIntegration.run_command(). The response is published
    back as an AgentResponseEvent for delivery.
    """

    def __init__(
        self,
        event_bus: EventBus,
        claude_integration: ClaudeIntegration,
        default_working_directory: Path,
        default_user_id: int = 0,
    ) -> None:
        self.event_bus = event_bus
        self.claude = claude_integration
        self.default_working_directory = default_working_directory
        self.default_user_id = default_user_id

    def register(self) -> None:
        """Subscribe to events that need agent processing."""
        self.event_bus.subscribe(WebhookEvent, self.handle_webhook)
        self.event_bus.subscribe(ScheduledEvent, self.handle_scheduled)

    async def handle_webhook(self, event: Event) -> None:
        """Process a webhook event through Claude.

        For GitHub events, only acts on ``issues`` and ``pull_request`` events
        where the action is ``labeled`` and the label is ``waiting-for-ai``.
        All other events are silently ignored (the HTTP server already returned
        200 to GitHub).

        On success, Claude's response is posted back as a GitHub comment and
        the labels are swapped (waiting-for-ai → waiting-for-human).
        On failure, an error comment is posted without touching labels so the
        developer can see what happened and retry.
        """
        if not isinstance(event, WebhookEvent):
            return

        # --- GitHub label filter ---
        if event.provider == "github":
            action = event.payload.get("action")
            label_name = event.payload.get("label", {}).get("name", "")
            event_type = event.event_type_name

            if not (
                event_type in ("issues", "pull_request")
                and action == "labeled"
                and label_name in _TRIGGER_LABELS
            ):
                logger.debug(
                    "Ignoring non-actionable GitHub webhook",
                    event_type=event_type,
                    action=action,
                    label=label_name,
                    delivery_id=event.delivery_id,
                )
                return

            # Extract repo + number for the response step
            gh_repo: Optional[str] = (
                event.payload.get("repository", {}).get("full_name")
            )
            issue_or_pr = event.payload.get("issue") or event.payload.get("pull_request") or {}
            gh_number: Optional[int] = issue_or_pr.get("number")
            gh_kind = "pr" if event_type == "pull_request" else "issue"
            # action-ready is for issues only — PRs use waiting-for-ai for all work
            is_action_ready = label_name == LABEL_ACTION_READY and gh_kind == "issue"

            logger.info(
                "Processing GitHub label event",
                repo=gh_repo,
                number=gh_number,
                kind=gh_kind,
                label=label_name,
                delivery_id=event.delivery_id,
            )

            prompt = self._build_github_prompt(event, action_ready=is_action_ready)

            # Notify Telegram that work is starting, before Claude runs
            issue_title = issue_or_pr.get("title", "")
            await self.event_bus.publish(
                AgentResponseEvent(
                    chat_id=0,
                    text=f'⏳ Picked up {gh_repo}#{gh_number} "{issue_title}" — working on it...',
                    originating_event_id=event.id,
                )
            )

            try:
                response = await self.claude.run_command(
                    prompt=prompt,
                    working_directory=self.default_working_directory,
                    user_id=self.default_user_id,
                )

                if response.content and gh_repo and gh_number:
                    await post_comment_and_remove_label(
                        repo=gh_repo,
                        number=gh_number,
                        kind=gh_kind,
                        response_text=response.content,
                        trigger_label=label_name,
                    )
                    # Also notify via Telegram if chat IDs are configured
                    await self.event_bus.publish(
                        AgentResponseEvent(
                            chat_id=0,
                            text=response.content,
                            originating_event_id=event.id,
                        )
                    )
            except Exception as exc:
                logger.exception(
                    "Agent execution failed for GitHub webhook event",
                    repo=gh_repo,
                    number=gh_number,
                    event_id=event.id,
                )
                if gh_repo and gh_number:
                    await post_error_comment(
                        repo=gh_repo,
                        number=gh_number,
                        error=str(exc),
                        trigger_label=label_name,
                    )
            return

        # --- Non-GitHub generic webhook (original behaviour) ---
        logger.info(
            "Processing generic webhook event through agent",
            provider=event.provider,
            event_type=event.event_type_name,
            delivery_id=event.delivery_id,
        )

        prompt = self._build_webhook_prompt(event)

        try:
            response = await self.claude.run_command(
                prompt=prompt,
                working_directory=self.default_working_directory,
                user_id=self.default_user_id,
            )

            if response.content:
                # We don't know which chat to send to from a webhook alone.
                # The notification service needs configured target chats.
                # Publish with chat_id=0 — the NotificationService
                # will broadcast to configured notification_chat_ids.
                await self.event_bus.publish(
                    AgentResponseEvent(
                        chat_id=0,
                        text=response.content,
                        originating_event_id=event.id,
                    )
                )
        except Exception:
            logger.exception(
                "Agent execution failed for webhook event",
                provider=event.provider,
                event_id=event.id,
            )

    async def handle_scheduled(self, event: Event) -> None:
        """Process a scheduled event through Claude."""
        if not isinstance(event, ScheduledEvent):
            return

        logger.info(
            "Processing scheduled event through agent",
            job_id=event.job_id,
            job_name=event.job_name,
        )

        prompt = event.prompt
        if event.skill_name:
            prompt = (
                f"/{event.skill_name}\n\n{prompt}" if prompt else f"/{event.skill_name}"
            )

        working_dir = event.working_directory or self.default_working_directory

        try:
            response = await self.claude.run_command(
                prompt=prompt,
                working_directory=working_dir,
                user_id=self.default_user_id,
            )

            if response.content:
                for chat_id in event.target_chat_ids:
                    await self.event_bus.publish(
                        AgentResponseEvent(
                            chat_id=chat_id,
                            text=response.content,
                            originating_event_id=event.id,
                        )
                    )

                # Also broadcast to default chats if no targets specified
                if not event.target_chat_ids:
                    await self.event_bus.publish(
                        AgentResponseEvent(
                            chat_id=0,
                            text=response.content,
                            originating_event_id=event.id,
                        )
                    )
        except Exception:
            logger.exception(
                "Agent execution failed for scheduled event",
                job_id=event.job_id,
                event_id=event.id,
            )

    def _build_github_prompt(self, event: WebhookEvent, action_ready: bool = False) -> str:
        """Build a rich prompt for a GitHub issue or PR label event.

        ``waiting-for-ai`` → discussion mode: read, respond, ask questions.
        ``action-ready``   → implementation mode: implement the task, raise a PR.
        """
        payload = event.payload
        repo = payload.get("repository", {}).get("full_name", "unknown/repo")
        event_type = event.event_type_name

        if event_type == "pull_request":
            pr = payload.get("pull_request", {})
            head = pr.get("head", {}).get("ref", "?")
            base = pr.get("base", {}).get("ref", "?")
            label = LABEL_ACTION_READY if action_ready else LABEL_WAITING_FOR_AI
            return (
                f"A GitHub pull request has been labelled `{label}` "
                f"and needs your attention.\n\n"
                f"Repository: {repo}\n"
                f"PR #{pr.get('number')}: {pr.get('title')}  [{head} → {base}]\n"
                f"URL: {pr.get('html_url')}\n\n"
                f"Description:\n{pr.get('body') or '(no description)'}\n\n"
                f"Please review any open review comments on the PR. Make any requested "
                f"code changes, push them to the branch, and post a summary comment on "
                f"the PR when done using `gh pr comment`. Do not merge the PR yourself."
            )
        else:
            issue = payload.get("issue", {})
            if action_ready:
                instruction = (
                    f"This issue is marked `{LABEL_ACTION_READY}` — implement the task "
                    f"described. Clone the repo if needed, make the changes, and raise a PR. "
                    f"Post a comment on the issue linking the PR when done."
                )
            else:
                instruction = (
                    f"This issue is marked `{LABEL_WAITING_FOR_AI}` — read it carefully and "
                    f"respond with a comment. You may ask clarifying questions or provide your "
                    f"analysis, but do NOT start implementing. Post your response using "
                    f"`gh issue comment` and nothing else."
                )
            label = LABEL_ACTION_READY if action_ready else LABEL_WAITING_FOR_AI
            return (
                f"A GitHub issue has been labelled `{label}` and needs your attention.\n\n"
                f"Repository: {repo}\n"
                f"Issue #{issue.get('number')}: {issue.get('title')}\n"
                f"URL: {issue.get('html_url')}\n\n"
                f"Body:\n{issue.get('body') or '(no description)'}\n\n"
                f"{instruction}"
            )

    def _build_webhook_prompt(self, event: WebhookEvent) -> str:
        """Build a Claude prompt from a generic (non-GitHub) webhook event."""
        payload_summary = self._summarize_payload(event.payload)

        return (
            f"A {event.provider} webhook event occurred.\n"
            f"Event type: {event.event_type_name}\n"
            f"Payload summary:\n{payload_summary}\n\n"
            f"Analyze this event and provide a concise summary. "
            f"Highlight anything that needs my attention."
        )

    def _summarize_payload(self, payload: Dict[str, Any], max_depth: int = 2) -> str:
        """Create a readable summary of a webhook payload."""
        lines: List[str] = []
        self._flatten_dict(payload, lines, max_depth=max_depth)
        # Cap at 2000 chars to keep prompt reasonable
        summary = "\n".join(lines)
        if len(summary) > 2000:
            summary = summary[:2000] + "\n... (truncated)"
        return summary

    def _flatten_dict(
        self,
        data: Any,
        lines: list,
        prefix: str = "",
        depth: int = 0,
        max_depth: int = 2,
    ) -> None:
        """Flatten a nested dict into key: value lines."""
        if depth >= max_depth:
            lines.append(f"{prefix}: ...")
            return

        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    self._flatten_dict(value, lines, full_key, depth + 1, max_depth)
                else:
                    val_str = str(value)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    lines.append(f"{full_key}: {val_str}")
        elif isinstance(data, list):
            lines.append(f"{prefix}: [{len(data)} items]")
            for i, item in enumerate(data[:3]):  # Show first 3 items
                self._flatten_dict(item, lines, f"{prefix}[{i}]", depth + 1, max_depth)
        else:
            lines.append(f"{prefix}: {data}")
