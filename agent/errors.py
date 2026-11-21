"""Failures of the agent backend itself, as opposed to wrong agent behaviour."""


class BackendError(RuntimeError):
    """The model API could not produce an answer. `kind` says why, e.g. "rate_limit".

    The eval runner records these as failures tagged `[external:<kind>]`, so an unreachable or
    throttled API is never mistaken for a passing run or for a bug in the agent.
    """

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
