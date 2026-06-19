"""
Week 6: Access Control, Rate Limiting & Cost Enforcement

Three guardrails that wrap the Week 5 agent and run BEFORE / AFTER the LLM:

1. AccessController - role-based document/field access control + response redaction
2. RateLimiter     - sliding-window limit on queries per user per minute
3. CostEnforcer    - per-role monthly budget enforcement

Design note: guardrails should block or sanitize a request as early as possible.
RateLimiter and CostEnforcer run *before* the LLM is ever called (cheap, prevent
abuse/runaway spend). AccessController.redact_response runs *after* the LLM
answers, scrubbing any sensitive values the caller's role isn't allowed to see.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve data files relative to THIS file so it runs from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# TASK 1: AccessController
# ============================================================================


class AccessController:
    """Enforce role-based access control."""

    def __init__(self, access_policy_path: str):
        """Load the access-control policy JSON into memory.

        `access_policy_path` may be relative; we resolve it against this file's
        directory so the agent works regardless of the current working dir.
        """
        if not os.path.isabs(access_policy_path):
            access_policy_path = os.path.join(SCRIPT_DIR, access_policy_path)
        with open(access_policy_path) as f:
            self.policy = json.load(f)

        # Convenience handles into the policy.
        self.document_access = self.policy.get("document_access", {})
        self.sensitive_fields = self.policy.get("sensitive_fields", {})

        # In-memory audit trail of every access decision.
        self.audit_log: List[Dict[str, Any]] = []

    # -- document-level access ----------------------------------------------

    def can_view_document(self, role: str, document: Dict[str, Any]) -> bool:
        """Return True if `role` is allowed to view `document`.

        Decision is based on the document's sensitivity level vs. the
        `document_access` map in the policy. Sensitivity is matched
        case-insensitively. Unknown / missing sensitivity is denied by default
        (fail closed).
        """
        sensitivity = (document.get("sensitivity") or "").strip()
        if not sensitivity:
            return False

        # Case-insensitive lookup against the policy's sensitivity keys.
        allowed_roles = None
        for level, roles in self.document_access.items():
            if level.lower() == sensitivity.lower():
                allowed_roles = roles
                break

        if allowed_roles is None:
            return False  # unknown sensitivity -> fail closed
        return role in allowed_roles

    # -- field-level access -------------------------------------------------

    def can_view_field(self, role: str, field_name: str) -> bool:
        """Return True if `role` may see the value of `field_name`.

        A field that isn't listed in `sensitive_fields` is not sensitive, so
        anyone can view it. A listed field is visible only to the roles in its
        `visibility` list.
        """
        field = self.sensitive_fields.get(field_name)
        if field is None:
            return True  # not a sensitive field -> visible to all
        return role in field.get("visibility", [])

    # -- response redaction --------------------------------------------------

    # Regex describing the *value* that follows a sensitive field's label.
    _VALUE_PATTERNS = {
        "salary": r"\$?\s?\d[\d,]*(?:\.\d{1,2})?\b",
        "compensation": r"\$?\s?\d[\d,]*(?:\.\d{1,2})?\b",
        "ssn": r"\d{3}-\d{2}-\d{4}",
        "address": r".+?(?=[.\n]|$)",
        "performance_review": r".+?(?=[.\n]|$)",
    }

    # Label/synonym words that introduce a sensitive value in free text.
    _LABEL_ALIASES = {
        "salary": ["base salary", "annual salary", "salary", "is paid", "earns",
                   "makes", "paid"],
        "compensation": ["total compensation", "compensation", "comp"],
        "ssn": ["social security number", "social security", "ssn"],
        "address": ["home address", "address"],
        "performance_review": ["performance review", "performance rating",
                               "review rating", "performance"],
    }

    def redact_response(self, role: str, response: str) -> str:
        """Replace any sensitive values `role` may not see with ``[REDACTED]``.

        For each sensitive field the role is NOT allowed to view, we look for
        the field's label (or a synonym) followed by its value, and replace just
        the value. SSNs are additionally scrubbed even when unlabeled, since
        their format is unambiguous and the policy marks them ``redact: true``.
        """
        if not response:
            return response

        redacted = response
        for field in self.sensitive_fields:
            if self.can_view_field(role, field):
                continue  # role is allowed to see this field

            self.log_access(role, "response", allowed=False, field=field)

            value_pat = self._VALUE_PATTERNS.get(field, r"\S+")
            for alias in self._LABEL_ALIASES.get(field, [field]):
                pattern = re.compile(
                    rf"({re.escape(alias)}\s*(?:is|:|=|of|->)?\s*)({value_pat})",
                    re.IGNORECASE,
                )
                redacted = pattern.sub(
                    lambda m: m.group(1) + "[REDACTED]", redacted
                )

        # Unconditional SSN scrub for roles that can't view SSNs.
        if not self.can_view_field(role, "ssn"):
            redacted = re.sub(r"\d{3}-\d{2}-\d{4}", "[REDACTED]", redacted)

        return redacted

    # -- audit ---------------------------------------------------------------

    def log_access(
        self, role: str, resource: str, allowed: bool, field: str = None
    ):
        """Append a timestamped entry to the audit log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "resource": resource,
            "field": field,
            "allowed": allowed,
        }
        self.audit_log.append(entry)
        logger.debug(f"AUDIT {entry}")

    def filter_documents(
        self, role: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return only the documents `role` is allowed to view (each decision
        is recorded in the audit log)."""
        visible = []
        for doc in documents:
            allowed = self.can_view_document(role, doc)
            self.log_access(
                role,
                resource=doc.get("id", doc.get("title", "document")),
                allowed=allowed,
            )
            if allowed:
                visible.append(doc)
        return visible

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log entries."""
        return self.audit_log


# ============================================================================
# TASK 2: RateLimiter
# ============================================================================


class RateLimiter:
    """Sliding-window rate limit: N queries per user per rolling 60 seconds."""

    def __init__(self, max_queries_per_minute: int = 30):
        self.max_queries_per_minute = max_queries_per_minute
        self.user_query_times: Dict[str, List[float]] = {}  # user_id -> [ts...]

    def _recent(self, user_id: str, now: float) -> List[float]:
        """Return (and store) this user's query timestamps from the last 60s."""
        window = [t for t in self.user_query_times.get(user_id, []) if now - t < 60]
        self.user_query_times[user_id] = window
        return window

    def is_allowed(self, user_id: str) -> bool:
        """Return True and record the query if the user is under the limit;
        otherwise return False without recording."""
        now = time()
        recent = self._recent(user_id, now)
        if len(recent) < self.max_queries_per_minute:
            recent.append(now)  # count this query
            return True
        return False

    def get_remaining_queries(self, user_id: str) -> int:
        """How many more queries the user may make in the current window."""
        now = time()
        recent = self._recent(user_id, now)
        return max(0, self.max_queries_per_minute - len(recent))


# ============================================================================
# TASK 3: CostEnforcer
# ============================================================================


class CostEnforcer:
    """Enforce per-role monthly budget limits (in USD)."""

    # Monthly budget per role.
    DEFAULT_BUDGETS = {
        "engineer": 100.0,
        "manager": 500.0,
        "hr": 200.0,
        "finance": 500.0,
        "executive": 1000.0,
    }
    # Used when a user's role is not yet known (e.g. a budget check before any
    # spend has been recorded and no role was supplied).
    _FALLBACK_ROLE = "engineer"

    def __init__(self, policy_path: str = None):
        """Load role budgets. If `policy_path` points to a JSON file with a
        `role_budgets` object we use that; otherwise we fall back to the
        documented defaults."""
        self.role_budgets = dict(self.DEFAULT_BUDGETS)
        if policy_path:
            try:
                path = policy_path
                if not os.path.isabs(path):
                    path = os.path.join(SCRIPT_DIR, path)
                with open(path) as f:
                    data = json.load(f)
                self.role_budgets.update(data.get("role_budgets", {}))
            except Exception as e:
                logger.warning(f"Could not load cost policy ({e}); using defaults")

        # user_id -> {"role": str, "total": float}
        self.user_spending: Dict[str, Dict[str, Any]] = {}

    def add_cost(self, user_id: str, role: str, cost: float):
        """Record `cost` spent by `user_id` (creating their record if needed)."""
        entry = self.user_spending.get(user_id)
        if entry is None:
            entry = {"role": role, "total": 0.0}
            self.user_spending[user_id] = entry
        if role:  # keep role current
            entry["role"] = role
        entry["total"] += cost

    def _budget_and_spent(self, user_id: str, role: str = None) -> tuple:
        """Resolve (budget, spent) for a user.

        If the user has spending history we use their recorded role; otherwise
        we use the `role` argument, falling back to the engineer budget.
        """
        entry = self.user_spending.get(user_id)
        if entry is not None:
            role_for_budget = entry["role"]
            spent = entry["total"]
        else:
            role_for_budget = role or self._FALLBACK_ROLE
            spent = 0.0
        budget = self.role_budgets.get(role_for_budget, 0.0)
        return budget, spent

    def can_afford_query(
        self, user_id: str, estimated_cost: float, role: str = None
    ) -> bool:
        """Return True if `estimated_cost` fits in the user's remaining budget."""
        budget, spent = self._budget_and_spent(user_id, role)
        remaining = budget - spent
        return estimated_cost <= remaining

    def get_budget_remaining(self, user_id: str, role: str = None) -> float:
        """Remaining budget for `user_id` (never negative)."""
        budget, spent = self._budget_and_spent(user_id, role)
        return max(0.0, budget - spent)


# ============================================================================
# TASK 5: Tests
# ============================================================================

if __name__ == "__main__":
    """Quick test of access-control functionality."""

    # --- AccessController --------------------------------------------------
    print("Testing AccessController...")
    controller = AccessController("data/access_control.json")

    assert not controller.can_view_field(
        "engineer", "salary"
    ), "Engineer should not see salary"
    assert controller.can_view_field("hr", "salary"), "HR should see salary"
    assert controller.can_view_field("manager", "salary"), "Manager should see salary"
    assert not controller.can_view_field(
        "engineer", "ssn"
    ), "Engineer should not see SSN"
    print("  can_view_field: PASSED")

    docs = [
        {"id": "doc1", "sensitivity": "Public", "content": "Mission statement"},
        {"id": "doc2", "sensitivity": "Confidential", "content": "Salary ranges"},
    ]
    visible = controller.filter_documents("engineer", docs)
    assert (
        len(visible) == 1 and visible[0]["id"] == "doc1"
    ), "Engineer should only see Public doc"
    print("  filter_documents: PASSED")

    # Bonus: redaction sanity check.
    answer = "Sarah's base salary is $185,000 and her SSN is 123-45-6789."
    redacted_eng = controller.redact_response("engineer", answer)
    assert "185,000" not in redacted_eng and "123-45-6789" not in redacted_eng, (
        "Engineer answer should be redacted"
    )
    assert "[REDACTED]" in redacted_eng, "Redaction marker should appear"
    full_hr = controller.redact_response("hr", answer)
    assert "185,000" in full_hr and "123-45-6789" in full_hr, (
        "HR should see full salary + SSN"
    )
    print("  redact_response: PASSED")

    # --- RateLimiter -------------------------------------------------------
    print("\nTesting RateLimiter...")
    limiter = RateLimiter(max_queries_per_minute=3)
    assert limiter.is_allowed("user1"), "First query should be allowed"
    assert limiter.is_allowed("user1"), "Second query should be allowed"
    assert limiter.is_allowed("user1"), "Third query should be allowed"
    assert not limiter.is_allowed("user1"), "Fourth query should be blocked"
    assert limiter.get_remaining_queries("user1") == 0, "No queries remaining"
    assert limiter.is_allowed("user2"), "Different user has its own budget"
    print("  is_allowed: PASSED")

    # --- CostEnforcer ------------------------------------------------------
    print("\nTesting CostEnforcer...")
    enforcer = CostEnforcer()
    assert enforcer.can_afford_query(
        "user1", 50.0
    ), "Should afford $50 within $100 budget"
    enforcer.add_cost("user1", "engineer", 50.0)
    assert enforcer.can_afford_query(
        "user1", 49.0
    ), "Should afford $49 with $50 remaining"
    assert not enforcer.can_afford_query(
        "user1", 51.0
    ), "Should not afford $51 with $50 remaining"
    assert enforcer.get_budget_remaining("user1") == 50.0, "Remaining should be $50"
    # Executive has a larger budget.
    enforcer.add_cost("exec1", "executive", 600.0)
    assert enforcer.can_afford_query("exec1", 300.0), "Executive has $1000 budget"
    print("  can_afford_query: PASSED")

    print("\nAll tests passed!")
