"""Claude-specific exceptions."""


class ClaudeError(Exception):
    """Base Claude error."""


class ClaudeTimeoutError(ClaudeError):
    """Operation timed out."""


class ClaudeProcessError(ClaudeError):
    """Process execution failed."""


class ClaudeRateLimitError(ClaudeError):
    """Claude API rate limit (HTTP 429 / overloaded) was hit.

    Raised instead of ClaudeProcessError when the underlying error is
    clearly a transient capacity/quota issue rather than a code bug,
    so callers can route the triggering event to the persistent retry
    queue instead of reporting it as a hard failure.
    """


class ClaudeParsingError(ClaudeError):
    """Failed to parse output."""


class ClaudeSessionError(ClaudeError):
    """Session management error."""


class ClaudeMCPError(ClaudeError):
    """MCP server connection or configuration error."""

    def __init__(self, message: str, server_name: str = None):
        super().__init__(message)
        self.server_name = server_name
