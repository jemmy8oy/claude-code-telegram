"""GitHub label-based workflow helpers.

Provides utilities for the waiting-for-ai / waiting-for-human handoff pattern:
- Post Claude's response as a comment on the originating issue or PR
- Swap the waiting-for-ai label to waiting-for-human on success
- Post a visible error comment (without swapping labels) on failure so the
  developer knows a retry is needed
"""

import asyncio
from typing import Literal

import structlog

logger = structlog.get_logger()

LABEL_WAITING_FOR_AI = "waiting-for-ai"
LABEL_WAITING_FOR_HUMAN = "waiting-for-human"


async def post_comment_and_swap_labels(
    repo: str,
    number: int,
    kind: Literal["issue", "pr"],
    response_text: str,
) -> None:
    """Post Claude's response as a comment and swap labels.

    Removes ``waiting-for-ai`` and applies ``waiting-for-human`` so the
    developer knows the ball is back in their court.

    Args:
        repo: Full repository name, e.g. ``"jemmy8oy/web-template"``.
        number: Issue or PR number.
        kind: ``"issue"`` or ``"pr"``.
        response_text: The text to post as a GitHub comment.
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
            "--remove-label", LABEL_WAITING_FOR_AI,
            "--add-label", LABEL_WAITING_FOR_HUMAN,
        ],
        context=f"{repo}#{number} label swap",
    )

    logger.info(
        "Posted comment and swapped labels",
        repo=repo,
        number=number,
        kind=kind,
    )


async def post_error_comment(
    repo: str,
    number: int,
    error: str,
) -> None:
    """Post a visible error comment without touching labels.

    Leaves ``waiting-for-ai`` in place (or lets the developer decide what to
    do), and makes the failure visible on the issue/PR so nothing silently
    disappears.

    Args:
        repo: Full repository name.
        number: Issue or PR number.
        error: Short error description to include in the comment.
    """
    body = (
        f"⚠️ **Claude encountered an error** and could not complete this task.\n\n"
        f"```\n{error}\n```\n\n"
        f"Please review the logs, then re-apply `{LABEL_WAITING_FOR_AI}` when ready to retry."
    )
    await _run(
        ["gh", "issue", "comment", str(number), "--repo", repo, "--body", body],
        context=f"{repo}#{number} error comment",
    )
    logger.warning("Posted error comment", repo=repo, number=number, error=error)


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
