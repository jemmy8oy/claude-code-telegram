"""Persistent event queue for rate-limit retry.

When the Claude API rate-limits a request the triggering GitHub webhook event
is written to this queue so it survives pod restarts and is retried when
capacity recovers.

Items are keyed by ``(repo, number)`` — INSERT OR REPLACE semantics mean that
if the same issue/PR fires a second webhook before the first has been replayed,
only the latest payload is kept.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import structlog

from .database import DatabaseManager

logger = structlog.get_logger()


class QueuedEventRepository:
    """CRUD operations for the ``queued_events`` table."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    async def enqueue(
        self,
        repo: str,
        number: int,
        kind: str,
        label: str,
        payload: Dict[str, Any],
    ) -> None:
        """Insert or replace a queued event.

        ``INSERT OR REPLACE`` ensures that if the same (repo, number) already
        exists, the row is overwritten with the freshest payload and the retry
        counter is reset to 0 — "latest wins".

        Args:
            repo:    Full repository name, e.g. ``"owner/repo"``.
            number:  Issue or PR number.
            kind:    ``"issue"`` or ``"pull_request"``.
            label:   The trigger label name (e.g. ``"waiting-for-ai"``).
            payload: The raw ``WebhookEvent.payload`` dict.
        """
        payload_json = json.dumps(payload)
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO queued_events
                    (repo, number, kind, label, payload, queued_at, retry_count, last_attempted_at)
                VALUES
                    (?, ?, ?, ?, ?, datetime('now'), 0, NULL)
                """,
                (repo, number, kind, label, payload_json),
            )
            await conn.commit()

        logger.info(
            "Event queued for rate-limit retry",
            repo=repo,
            number=number,
            kind=kind,
            label=label,
        )

    async def list_pending(
        self,
        max_retries: int,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return queued items that have not yet exhausted their retry budget.

        Ordered oldest-first so that events that have been waiting longest are
        attempted first.

        Args:
            max_retries: Items with ``retry_count >= max_retries`` are excluded.
            limit:       Maximum number of rows to return per call.

        Returns:
            List of row dicts with keys:
            ``id, repo, number, kind, label, payload, queued_at,
              retry_count, last_attempted_at``.
            ``payload`` is returned as a parsed ``dict``.
        """
        async with self.db_manager.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, repo, number, kind, label, payload,
                       queued_at, retry_count, last_attempted_at
                FROM   queued_events
                WHERE  retry_count < ?
                ORDER  BY queued_at ASC
                LIMIT  ?
                """,
                (max_retries, limit),
            )
            rows = await cursor.fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["payload"] = json.loads(d["payload"])
            result.append(d)
        return result

    async def mark_attempted(self, item_id: int) -> None:
        """Increment retry_count and update last_attempted_at for an item.

        Called when a drain attempt fails (rate limit still active or other
        error) so the next drain run sees the updated counter.

        Args:
            item_id: Primary-key ``id`` of the queued_events row.
        """
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                UPDATE queued_events
                SET    retry_count      = retry_count + 1,
                       last_attempted_at = datetime('now')
                WHERE  id = ?
                """,
                (item_id,),
            )
            await conn.commit()

    async def dequeue(self, item_id: int) -> None:
        """Permanently remove an item from the queue (success or abandonment).

        Args:
            item_id: Primary-key ``id`` of the queued_events row.
        """
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                "DELETE FROM queued_events WHERE id = ?",
                (item_id,),
            )
            await conn.commit()

        logger.debug("Event dequeued", item_id=item_id)

    async def count_pending(self, max_retries: int) -> int:
        """Return the number of items that still have retry budget remaining.

        Args:
            max_retries: Items with ``retry_count >= max_retries`` are excluded.

        Returns:
            Count of actionable queue entries.
        """
        async with self.db_manager.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM queued_events WHERE retry_count < ?",
                (max_retries,),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0
