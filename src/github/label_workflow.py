"""GitHub label-based workflow helpers.

Provides utilities for the waiting-for-ai / action-ready handoff pattern:
- Remove waiting-for-ai or action-ready on success (no waiting-for-human —
  anything without a trigger label is implicitly the human's turn)
- Post Claude's response as a comment on the originating issue or PR
  (kept for legacy / testing use; the preferred flow is for Claude to post
  its own comment via ``gh issue comment`` and the system only removes the
  label via ``remove_trigger_label``).
- Post a visible error comment and apply ai-error on failure so the
  developer knows a retry is needed
"""

import asyncio
from typing import Literal

import structlog

logger = structlog.get_logger()

LABEL_WAITING_FOR_AI = "waiting-for-ai"
LABEL_ACTION_READY = "action-ready"
LABEL_AI_ERROR = "ai-error"


async def post_comment_and_remove_label(
    repo: str,
    number: int,
    kind: Literal["issue", "pr"],
    response_text: str,
    trigger_label: str = LABEL_WAITING_FOR_AI,
) -> None:
    """Post Claude's response as a comment and remove the trigger label.

    No replacement label is applied — anything without a trigger label is
    implicitly the human's turn.

    Args:
        repo: Full repository name, e.g. ``"jemmy8oy/web-template"``.
        number: Issue or PR number.
        kind: ``"issue"`` or ``"pr"``.
        response_text: The text to post as a GitHub comment.
        trigger_label: The label to remove (waiting-for-ai or action-ready).
    """
    comment_cmd = (
        ["gh", "issue", "comment", str(number), "--repo", repo, "--body", response_text]
        if kind == "issue"
        else ["gh", "pr", "comment", str(number), "--repo", repo, "--body", response_text]
    )

    await _run(comment_cmd, context=f"{repo}#{number} comment")

    # gh issue edit works for both issues and PRs for label management
    await _run(
        [
            "gh", "issue", "edit", str(number),
            "--repo", repo,
            "--remove-label", trigger_label,
        ],
        context=f"{repo}#{number} remove {trigger_label}",
    )

    logger.info(
        "Posted comment and removed trigger label",
        repo=repo,
        number=number,
        kind=kind,
        trigger_label=trigger_label,
    )


async def remove_trigger_label(
    repo: str,
    number: int,
    trigger_label: str = LABEL_WAITING_FOR_AI,
) -> None:
    """Remove the trigger label from an issue or PR without posting a comment.

    Used when Claude has already posted its own response via ``gh issue comment``
    or ``gh pr comment`` and the system only needs to clear the label so the
    issue returns to the human's queue.

    Args:
        repo: Full repository name, e.g. ``"jemmy8oy/web-template"``.
        number: Issue or PR number.
        trigger_label: The label to remove (waiting-for-ai or action-ready).
    """
    # gh issue edit works for both issues and PRs for label management
    await _run(
        [
            "gh", "issue", "edit", str(number),
            "--repo", repo,
            "--remove-label", trigger_label,
        ],
        context=f"{repo}#{number} remove {trigger_label}",
    )

    logger.info(
        "Removed trigger label",
        repo=repo,
        number=number,
        trigger_label=trigger_label,
    )


async def post_error_comment(
    repo: str,
    number: int,
    error: str,
    trigger_label: str = LABEL_WAITING_FOR_AI,
) -> None:
    """Post a visible error comment and apply the ai-error label.

    Removes the trigger label and applies ``ai-error`` so the failure is
    visible at a glance in the issue list and the bot won't re-trigger.

    Args:
        repo: Full repository name.
        number: Issue or PR number.
        error: Short error description to include in the comment.
        trigger_label: The label that triggered this run (to be removed).
    """
    body = (
        f"⚠️ **Claude encountered an error** and could not complete this task.\n\n"
        f"```\n{error}\n```\n\n"
        f"Please review the logs, then re-apply `{trigger_label}` when ready to retry."
    )
    await _run(
        ["gh", "issue", "comment", str(number), "--repo", repo, "--body", body],
        context=f"{repo}#{number} error comment",
    )
    await _run(
        [
            "gh", "issue", "edit", str(number),
            "--repo", repo,
            "--remove-label", trigger_label,
            "--add-label", LABEL_AI_ERROR,
        ],
        context=f"{repo}#{number} apply ai-error",
    )
    logger.warning("Posted error comment and applied ai-error", repo=repo, number=number, error=error)


async def _run(cmd: list, context: str = "") -> None:
    """Run a shell command asynchronously, logging any failures."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(
            "gh command failed",
            context=context,
            returncode=proc.returncode,
            stderr=stderr.decode("utf-8", errors="replace").strip(),
        )
    else:
        logger.debug(
            "gh command succeeded",
            context=context,
            stdout=stdout.decode("utf-8", errors="replace").strip(),
        )
