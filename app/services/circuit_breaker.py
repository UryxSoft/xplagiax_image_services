"""
Distributed Circuit Breaker backed by Redis.
Protects external APIs from cascading failures and coordinates state across Gunicorn workers.
"""

import time
from typing import Optional
from redis import Redis

from app.observability.telemetry import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    def __init__(
        self,
        redis_client: Optional[Redis],
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_secs: int = 60,
    ):
        self.redis = redis_client
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_secs = recovery_timeout_secs
        
        self._key_failures = f"cb:failures:{self.name}"
        self._key_state = f"cb:state:{self.name}"

    def is_allowed(self) -> bool:
        """Check if the circuit is CLOSED (allowed) or HALF-OPEN."""
        if not self.redis:
            return True  # Degrade gracefully if no Redis

        state = self.redis.get(self._key_state)
        if state == b"OPEN":
            return False
        return True

    def record_failure(self):
        """Record a failure and potentially trip the circuit."""
        if not self.redis:
            return

        failures = self.redis.incr(self._key_failures)
        if failures == 1:
            self.redis.expire(self._key_failures, self.recovery_timeout_secs)

        if failures >= self.failure_threshold:
            self._trip_circuit()

    def record_success(self):
        """Reset failures on success."""
        if not self.redis:
            return
        
        # If it was HALF-OPEN or OPEN, reset it
        if self.redis.get(self._key_state):
            self.redis.delete(self._key_state)
            self.redis.delete(self._key_failures)
            logger.info(f"Circuit {self.name} CLOSED (Recovered).")

    def _trip_circuit(self):
        """Transition state to OPEN."""
        is_already_open = self.redis.set(
            self._key_state, "OPEN", ex=self.recovery_timeout_secs, nx=True
        )
        if is_already_open:
            logger.error(f"Circuit {self.name} tripped to OPEN state for {self.recovery_timeout_secs}s.")
