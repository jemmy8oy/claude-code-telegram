"""Event handlers that bridge the event bus to Claude and Telegram.

AgentHandler: translates events into ClaudeIntegration.run_command() calls.
NotificationHandler: subscribes to AgentResponseEvent and delivers to Telegram.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..claude.exceptions import ClaudeRateLimitError
from ..claude.facade import ClaudeIntegration
from ..github.label_workflow import (
    LABEL_ACTION_READY,
    LABEL_WAITING_FOR_AI,
    post_error_comment,
    remove_trigger_label,
)
from ..storage.queue_repository import QueuedEventRepository
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

    When the Claude API returns a rate-limit error the triggering GitHub
    event is written to the persistent ``queued_events`` table (via
    *queue_repo*) so it survives pod restarts.  A separate scheduled drain
    job calls :meth:`drain_persistent_queue` periodically to replay those
    items once capacity has recovered.
    """

    def __init__(
        self,
        event_bus: EventBus,
        claude_integration: ClaudeIntegration,
        default_working_directory: Path,
        default_user_id: int = 0,
        queue_repo: Optional[QueuedEventRepository] = None,
    ) -> None:
        self.event_bus = event_bus
        self.claude = claude_integration
        self.default_working_directory = default_working_directory
        self.default_user_id = default_user_id
        self.queue_repo = queue_repo

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

            # Refresh GH_TOKEN before every task — keeps the agent's gh CLI
            # authenticated without relying on in-session recovery instructions.
            await self._refresh_gh_token()

            # Notify Telegram that work is starting, before Claude runs
            issue_title = issue_or_pr.get("title", "")
            kind_label = "PR" if gh_kind == "pr" else "issue"
            await self.event_bus.publish(
                AgentResponseEvent(
                    chat_id=0,
                    text=f'⏳ Picked up {kind_label} {gh_repo}#{gh_number} "{issue_title}" — working on it...',
                    originating_event_id=event.id,
                )
            )

            # Schedule Claude work as a background task so the EventBus
            # dispatch loop is not blocked during the long-running
            # run_command() call.  Returning here lets the event bus
            # immediately process and deliver the pickup notification
            # above before Claude even starts.
            asyncio.create_task(
                self._run_github_task(
                    event=event,
                    prompt=prompt,
                    gh_repo=gh_repo,
                    gh_number=gh_number,
                    gh_kind=gh_kind,
                    label_name=label_name,
                )
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

    async def _run_github_task(
        self,
        event: Event,
        prompt: str,
        gh_repo: Optional[str],
        gh_number: Optional[int],
        gh_kind: str,
        label_name: str,
    ) -> None:
        """Execute Claude for a GitHub webhook event in the background.

        Separated from handle_webhook() so that asyncio.create_task() can
        schedule this without blocking the EventBus dispatch loop.  This
        guarantees the pickup notification is delivered to Telegram before
        the long-running run_command() call begins.
        """
        try:
            response = await self.claude.run_command(
                prompt=prompt,
                working_directory=self.default_working_directory,
                user_id=self.default_user_id,
                force_new=True,
            )

            if gh_repo and gh_number:
                # Claude has already posted its own GitHub comment via
                # `gh issue comment` / `gh pr comment` as instructed in the
                # prompt.  We only need to remove the trigger label here so
                # the issue returns to the human's queue — posting a second
                # comment programmatically would cause double replies.
                await remove_trigger_label(
                    repo=gh_repo,
                    number=gh_number,
                    trigger_label=label_name,
                )
            # Notify Telegram with Claude's response summary
            if response.content:
                await self.event_bus.publish(
                    AgentResponseEvent(
                        chat_id=0,
                        text=response.content,
                        originating_event_id=event.id,
                    )
                )
        except ClaudeRateLimitError as exc:
            logger.warning(
                "Claude API rate limit hit — queuing event for retry",
                repo=gh_repo,
                number=gh_number,
                event_id=event.id,
                error=str(exc),
            )
            if self.queue_repo and gh_repo and gh_number:
                await self.queue_repo.enqueue(
                    repo=gh_repo,
                    number=gh_number,
                    kind=gh_kind,
                    label=label_name,
                    payload=event.payload,
                )
                await self.event_bus.publish(
                    AgentResponseEvent(
                        chat_id=0,
                        text=(
                            f"⏰ Claude rate limit hit — "
                            f"{gh_repo}#{gh_number} queued for automatic retry."
                        ),
                        originating_event_id=event.id,
                    )
                )
            else:
                # No queue configured — fall back to error comment so the
                # developer at least sees the failure.
                if gh_repo and gh_number:
                    await post_error_comment(
                        repo=gh_repo,
                        number=gh_number,
                        error=str(exc),
                        trigger_label=label_name,
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

    async def _refresh_gh_token(self) -> None:
        """Refresh GH_TOKEN via the refresh script before each webhook task.

        Tokens from GitHub App installations expire after one hour. Refreshing
        before every task ensures the agent's ``gh`` CLI calls succeed without
        needing in-session recovery instructions.
        """
        proc = await asyncio.create_subprocess_exec(
            "python3", "/data/workspace/refresh_gh.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            os.environ["GH_TOKEN"] = stdout.decode().strip()
            logger.debug("GH_TOKEN refreshed successfully")
        else:
            logger.warning(
                "GH_TOKEN refresh failed — proceeding with existing token",
                stderr=stderr.decode("utf-8", errors="replace").strip(),
            )

    # ------------------------------------------------------------------
    # Persistent queue drain
    # ------------------------------------------------------------------

    async def drain_persistent_queue(self, max_retries: int = 10) -> None:
        """Attempt to replay events that were queued due to a rate limit.

        Called by the APScheduler drain job at a configurable interval.

        Design:
        - Fetch pending items oldest-first (up to a reasonable batch).
        - For each item, re-verify the trigger label is still present.
        - Attempt ``run_command()``; on success dequeue and remove the label.
        - On :class:`~..claude.exceptions.ClaudeRateLimitError` the rate limit
          is still active — increment the retry counter and **stop the entire
          batch** (global retry: no point attempting remaining items).
        - After ``max_retries`` failed attempts the item is abandoned: the
          ``ai-error`` label is applied and a Telegram alert is sent.

        Args:
            max_retries: Abandon items whose ``retry_count`` reaches this value.
        """
        if not self.queue_repo:
            return

        pending = await self.queue_repo.list_pending(max_retries=max_retries)
        if not pending:
            logger.debug("Persistent queue is empty — nothing to drain")
            return

        logger.info("Draining persistent event queue", pending_count=len(pending))

        for item in pending:
            repo: str = item["repo"]
            number: int = item["number"]
            kind: str = item["kind"]
            label: str = item["label"]
            payload: Dict[str, Any] = item["payload"]
            item_id: int = item["id"]
            retry_count: int = item["retry_count"]

            # 1. Re-verify that the trigger label is still on the issue/PR.
            #    If it has been removed (human cancelled, already handled, etc.)
            #    silently drop the queued item.
            if not await self._label_still_present(repo, number, label):
                logger.info(
                    "Trigger label no longer present — dropping queued item",
                    repo=repo,
                    number=number,
                    label=label,
                )
                await self.queue_repo.dequeue(item_id)
                continue

            # 2. Refresh GH_TOKEN and attempt replay.
            await self._refresh_gh_token()

            # Reconstruct a minimal WebhookEvent so _build_github_prompt()
            # can derive the prompt from the original payload without needing
            # to re-fetch anything from GitHub.
            mock_event = WebhookEvent(
                provider="github",
                event_type_name="issues" if kind == "issue" else "pull_request",
                payload=payload,
            )
            is_action_ready = label == LABEL_ACTION_READY and kind == "issue"
            prompt = self._build_github_prompt(mock_event, action_ready=is_action_ready)

            try:
                response = await self.claude.run_command(
                    prompt=prompt,
                    working_directory=self.default_working_directory,
                    user_id=self.default_user_id,
                    force_new=True,
                )

                # Success — remove from queue and clear the trigger label.
                await self.queue_repo.dequeue(item_id)
                await remove_trigger_label(
                    repo=repo,
                    number=number,
                    trigger_label=label,
                )
                logger.info(
                    "Queued event replayed successfully",
                    repo=repo,
                    number=number,
                    retry_count=retry_count,
                )
                if response.content:
                    await self.event_bus.publish(
                        AgentResponseEvent(
                            chat_id=0,
                            text=response.content,
                            originating_event_id=mock_event.id,
                        )
                    )

            except ClaudeRateLimitError:
                # Rate limit is still active.  Increment the counter and stop
                # the whole batch — no point trying remaining items.
                new_count = retry_count + 1
                await self.queue_repo.mark_attempted(item_id)
                logger.warning(
                    "Rate limit still active during drain — stopping batch",
                    repo=repo,
                    number=number,
                    retry_count=new_count,
                    max_retries=max_retries,
                )
                if new_count >= max_retries:
                    await self._abandon_queued_item(
                        item_id=item_id,
                        repo=repo,
                        number=number,
                        label=label,
                        retry_count=new_count,
                        mock_event_id=mock_event.id,
                    )
                break  # global stop — wait for the next scheduled interval

            except Exception as exc:
                # Genuine processing error (not a rate limit).  Increment and,
                # if exhausted, abandon with ai-error.
                new_count = retry_count + 1
                await self.queue_repo.mark_attempted(item_id)
                logger.exception(
                    "Error replaying queued event",
                    repo=repo,
                    number=number,
                    retry_count=new_count,
                )
                if new_count >= max_retries:
                    await self._abandon_queued_item(
                        item_id=item_id,
                        repo=repo,
                        number=number,
                        label=label,
                        retry_count=new_count,
                        mock_event_id=mock_event.id,
                        error=str(exc),
                    )

    async def _abandon_queued_item(
        self,
        item_id: int,
        repo: str,
        number: int,
        label: str,
        retry_count: int,
        mock_event_id: str,
        error: str = "rate limit retries exhausted",
    ) -> None:
        """Remove a queued item that has exhausted its retry budget.

        Applies ``ai-error``, posts an error comment, sends a Telegram alert,
        and deletes the row from the queue.

        Args:
            item_id:       DB primary key of the queued_events row.
            repo:          Full repository name.
            number:        Issue or PR number.
            label:         The trigger label (for the error comment).
            retry_count:   Final retry count (for the alert message).
            mock_event_id: ID from the synthetic WebhookEvent used for replay.
            error:         Short description to include in the GitHub comment.
        """
        await self.queue_repo.dequeue(item_id)
        await post_error_comment(
            repo=repo,
            number=number,
            error=f"Abandoned after {retry_count} retries: {error}",
            trigger_label=label,
        )
        await self.event_bus.publish(
            AgentResponseEvent(
                chat_id=0,
                text=(
                    f"⛔ {repo}#{number} abandoned after {retry_count} rate-limit "
                    f"retries — `ai-error` label applied."
                ),
                originating_event_id=mock_event_id,
            )
        )
        logger.error(
            "Queued event abandoned after max retries",
            repo=repo,
            number=number,
            retry_count=retry_count,
        )

    async def _label_still_present(
        self, repo: str, number: int, label: str
    ) -> bool:
        """Return True if *label* is currently applied to the issue/PR.

        Uses ``gh issue view --json labels`` which works for both issues and
        pull requests.  On any CLI error, returns ``True`` (fail-open: better
        to attempt a replay than to silently drop an item).

        Args:
            repo:   Full repository name, e.g. ``"owner/repo"``.
            number: Issue or PR number.
            label:  Label name to check for.
        """
        proc = await asyncio.create_subprocess_exec(
            "gh", "issue", "view", str(number), "--repo", repo, "--json", "labels",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "Could not check label presence — assuming present",
                repo=repo,
                number=number,
                label=label,
                stderr=stderr.decode("utf-8", errors="replace").strip(),
            )
            return True
        try:
            data = json.loads(stdout.decode())
            labels = [lbl["name"] for lbl in data.get("labels", [])]
            return label in labels
        except Exception:
            logger.warning(
                "Failed to parse label JSON — assuming label present",
                repo=repo,
                number=number,
                label=label,
            )
            return True

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
                f"Remember to run ```export GH_TOKEN=$(python3 /data/workspace/refresh_gh.py)``` if you need to reauth with gh."
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
                f"Remember to run ```export GH_TOKEN=$(python3 /data/workspace/refresh_gh.py)``` if you need to reauth with gh."
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
