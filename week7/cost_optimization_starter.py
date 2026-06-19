"""
Week 7: Cost Optimization & Feedback Loop

Three systems that make the agent cheaper and self-improving:
1. CostAnalyzer        - break query cost down by component; flag expensive spikes
2. OptimizationStrategy - caching, retrieval trimming, model selection, compression
3. FeedbackLoop        - collect/validate user corrections and measure their impact

All three are pure Python (no LLM calls), so the test block at the bottom runs
deterministically and offline.
"""

import re
import logging
import statistics
from collections import Counter
from typing import Dict, List, Any
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# TASK 1: CostAnalyzer
# ============================================================================


class CostAnalyzer:
    """Analyze and track query costs by component."""

    # Cost components we break every query down into.
    COMPONENTS = ("retrieval_cost", "llm_cost", "tool_cost", "error_cost")

    def __init__(self):
        self.query_history: List[Dict[str, Any]] = []

    def record_query(self, query: Dict[str, Any]):
        """Record a query and its per-component cost breakdown.

        Missing components default to 0; total_cost is computed if not supplied;
        a UTC timestamp is added if absent.
        """
        entry = {
            "query_text": query.get("query_text", ""),
            "retrieval_cost": float(query.get("retrieval_cost", 0.0)),
            "llm_cost": float(query.get("llm_cost", 0.0)),
            "tool_cost": float(query.get("tool_cost", 0.0)),
            "error_cost": float(query.get("error_cost", 0.0)),
        }
        entry["total_cost"] = float(
            query.get("total_cost", sum(entry[c] for c in self.COMPONENTS))
        )
        entry["timestamp"] = query.get("timestamp", _utc_now())
        self.query_history.append(entry)

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Totals for every cost component across all recorded queries."""
        totals = {c: 0.0 for c in self.COMPONENTS}
        for q in self.query_history:
            for c in self.COMPONENTS:
                totals[c] += q[c]
        total_daily = sum(totals.values())
        return {
            "retrieval_total": totals["retrieval_cost"],
            "llm_total": totals["llm_cost"],
            "tool_total": totals["tool_cost"],
            "error_total": totals["error_cost"],
            "total_daily": total_daily,
            "query_count": len(self.query_history),
        }

    def identify_cost_spikes(self) -> List[Dict]:
        """Return queries whose total cost is a statistical outlier.

        A query is a spike if its cost exceeds mean + 2*stdev of all query
        costs. Needs at least two queries (and non-zero spread) to flag anything.
        """
        costs = [q["total_cost"] for q in self.query_history]
        if len(costs) < 2:
            return []

        mean = statistics.mean(costs)
        stdev = statistics.stdev(costs)
        if stdev == 0:
            return []

        threshold = mean + 2 * stdev
        spikes = []
        for q in self.query_history:
            if q["total_cost"] > threshold:
                spike = dict(q)
                spike["threshold"] = threshold
                spike["times_mean"] = round(q["total_cost"] / mean, 1) if mean else 0
                spikes.append(spike)
        return spikes


# ============================================================================
# TASK 2: OptimizationStrategy
# ============================================================================


class OptimizationStrategy:
    """Optimize agent costs through multiple strategies."""

    # Rough per-strategy savings estimates (% of query cost).
    _SAVINGS = {
        "caching": 30,
        "retrieval_reduction": 20,
        "model_selection": 25,
        "compression": 10,
    }
    # Words that signal a query needs the stronger (pricier) model.
    _COMPLEX = {
        "analyze", "analyse", "compare", "design", "evaluate", "recommend",
        "optimize", "optimise", "architect", "summarize", "summarise",
        "explain why", "trade-off", "tradeoff", "strategy", "forecast",
    }
    _TOP_K = 3  # documents to keep after retrieval trimming

    def __init__(self):
        self.cache: Dict[str, str] = {}        # query -> response
        self.strategies_applied: List[str] = []

    def _mark(self, strategy: str):
        """Record that a strategy was used (once)."""
        if strategy not in self.strategies_applied:
            self.strategies_applied.append(strategy)

    def apply_caching(self, query: str, response: str) -> tuple:
        """Return (is_cache_hit, response). On a hit, the cached response is
        returned and the LLM call is avoided entirely."""
        if query in self.cache:
            self._mark("caching")
            return (True, self.cache[query])
        self.cache[query] = response
        return (False, response)

    def optimize_retrieval_count(self, num_docs: int) -> int:
        """Trim how many documents we retrieve to the top-k (default 3),
        which cuts the tokens sent to the LLM."""
        optimized = max(1, min(num_docs, self._TOP_K))
        if optimized < num_docs:
            self._mark("retrieval_reduction")
        return optimized

    def select_model_by_complexity(self, query: str) -> str:
        """Pick the cheapest model that can handle the query.

        - Complex (analyze/compare/design/...) -> gemini-2.5-pro
        - Simple (what is/who/define/list)      -> gemini-2.5-flash-lite
        - Everything else                       -> gemini-2.5-flash
        """
        self._mark("model_selection")
        q = query.lower()
        if any(word in q for word in self._COMPLEX):
            return "gemini-2.5-pro"
        if re.match(r"\s*(what is|what's|who|when|where|define|list)\b", q):
            return "gemini-2.5-flash-lite"
        return "gemini-2.5-flash"

    def enable_response_compression(self, response: str, max_sentences: int = 3) -> str:
        """Keep only the first `max_sentences` sentences of a long response."""
        sentences = re.split(r"(?<=[.!?])\s+", response.strip())
        if len(sentences) <= max_sentences:
            return response
        self._mark("compression")
        return " ".join(sentences[:max_sentences]).strip()

    def get_optimization_impact(self) -> Dict[str, Any]:
        """Estimated cost savings from the strategies actually applied."""
        breakdown = {
            s: self._SAVINGS[s] for s in self.strategies_applied if s in self._SAVINGS
        }
        # Diminishing returns: cap combined savings so it stays realistic.
        total = min(85, sum(breakdown.values()))
        return {
            "total_savings_pct": float(total),
            "strategies_applied": list(self.strategies_applied),
            "breakdown": breakdown,
        }


# ============================================================================
# TASK 3: FeedbackLoop
# ============================================================================


class FeedbackLoop:
    """Collect and validate user corrections for continuous improvement."""

    MIN_AUTHORITY = 3  # manager or above may correct the agent

    def __init__(self):
        self.corrections: List[Dict[str, Any]] = []
        # Authority hierarchy for role-based validation.
        self.authority = {
            "engineer": 1,
            "hr": 2,
            "finance": 2,
            "manager": 3,
            "executive": 4,
        }

    def _is_valid(self, user_role: str, original_answer: str, corrected_answer: str):
        """Shared validation: sufficient authority AND a more detailed answer.

        Returns (ok: bool, reason: str)."""
        level = self.authority.get(user_role, 0)
        if level < self.MIN_AUTHORITY:
            return False, (
                f"role '{user_role}' (level {level}) lacks authority; "
                f"manager+ (level {self.MIN_AUTHORITY}) required"
            )
        if len(corrected_answer.strip()) <= len(original_answer.strip()):
            return False, "correction must be more detailed than the original answer"
        return True, "accepted"

    def submit_correction(
        self,
        original_query: str,
        original_answer: str,
        corrected_answer: str,
        user_role: str,
    ) -> Dict[str, Any]:
        """Validate and store a correction. All submissions are stored (with a
        `valid` flag) so feedback metrics can measure quality over time."""
        ok, reason = self._is_valid(user_role, original_answer, corrected_answer)
        record = {
            "original_query": original_query,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "user_role": user_role,
            "authority_level": self.authority.get(user_role, 0),
            "valid": ok,
            "timestamp": _utc_now(),
        }
        self.corrections.append(record)
        return {"accepted": ok, "reason": reason}

    def validate_correction(self, index: int) -> bool:
        """Re-validate a stored correction by index."""
        if index < 0 or index >= len(self.corrections):
            return False
        c = self.corrections[index]
        ok, _ = self._is_valid(
            c["user_role"], c["original_answer"], c["corrected_answer"]
        )
        return ok

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Quality metrics over all corrections collected so far."""
        total = len(self.corrections)
        if total == 0:
            return {
                "total_corrections": 0,
                "validation_rate": 0.0,
                "avg_correction_length": 0.0,
                "top_error_patterns": [],
            }

        valid = sum(1 for c in self.corrections if c["valid"])
        avg_len = statistics.mean(len(c["corrected_answer"]) for c in self.corrections)

        # "Error patterns" = the recurring subjects of corrected queries, keyed
        # on the first few words of each original query.
        patterns = Counter(
            " ".join(c["original_query"].lower().split()[:4]) for c in self.corrections
        )
        top = [{"pattern": p, "count": n} for p, n in patterns.most_common(3)]

        return {
            "total_corrections": total,
            "validation_rate": round(100 * valid / total, 1),
            "avg_correction_length": round(avg_len, 1),
            "top_error_patterns": top,
        }


# ============================================================================
# TASK 4: Tests
# ============================================================================

if __name__ == "__main__":
    # ----------------------------------------------------------------- 1
    print("Testing CostAnalyzer...")
    analyzer = CostAnalyzer()
    # A realistic day of cheap, similar queries...
    normal_queries = [
        ("What is the travel policy?", 0.001, 0.0040, 0.0005),
        ("Expense limit for a manager?", 0.001, 0.0030, 0.0005),
        ("Look up employee Brian Yang", 0.000, 0.0045, 0.0010),
        ("What are the PTO rules?", 0.001, 0.0035, 0.0005),
        ("Who approves expenses?", 0.001, 0.0032, 0.0005),
        ("What is the remote work policy?", 0.001, 0.0038, 0.0005),
        ("Expense limit for a director?", 0.001, 0.0036, 0.0005),
        ("What is the parental leave policy?", 0.001, 0.0034, 0.0005),
    ]
    for text, retr, llm, tool in normal_queries:
        analyzer.record_query({
            "query_text": text, "retrieval_cost": retr,
            "llm_cost": llm, "tool_cost": tool,
        })
    # ...and one runaway query (heavy retrieval + big LLM cost + retries) -> spike.
    analyzer.record_query({
        "query_text": "Analyze every department budget for the year",
        "retrieval_cost": 0.02, "llm_cost": 0.08, "tool_cost": 0.005,
        "error_cost": 0.03,
    })

    breakdown = analyzer.get_cost_breakdown()
    print(f"  breakdown: {breakdown}")
    assert breakdown["query_count"] == 9
    assert round(breakdown["total_daily"], 4) == round(
        sum(q["total_cost"] for q in analyzer.query_history), 4
    )
    print("  get_cost_breakdown: PASSED")

    spikes = analyzer.identify_cost_spikes()
    print(f"  spikes detected: {[s['query_text'] for s in spikes]}")
    assert len(spikes) == 1 and spikes[0]["query_text"].startswith("Analyze")
    print("  identify_cost_spikes: PASSED")

    # ----------------------------------------------------------------- 2
    print("\nTesting OptimizationStrategy...")
    optimizer = OptimizationStrategy()

    miss = optimizer.apply_caching("What is the travel policy?", "Pre-approval needed.")
    hit = optimizer.apply_caching("What is the travel policy?", "Pre-approval needed.")
    print(f"  first call (miss): {miss}")
    print(f"  second call (hit): {hit}")
    assert miss == (False, "Pre-approval needed.")
    assert hit == (True, "Pre-approval needed.")
    print("  apply_caching: PASSED")

    assert optimizer.optimize_retrieval_count(15) == 3
    print(f"  optimize_retrieval_count(15) -> {optimizer.optimize_retrieval_count(15)}: PASSED")

    simple = optimizer.select_model_by_complexity("What is the travel policy?")
    complex_ = optimizer.select_model_by_complexity("Analyze and compare Q3 budgets")
    print(f"  simple  query -> {simple}")
    print(f"  complex query -> {complex_}")
    assert simple == "gemini-2.5-flash-lite"
    assert complex_ == "gemini-2.5-pro"
    print("  select_model_by_complexity: PASSED")

    long_resp = ("Travel must be pre-approved. Domestic limit is $5000. "
                 "International requires director sign-off. Receipts due in 30 days. "
                 "Reimbursement takes two weeks.")
    compressed = optimizer.enable_response_compression(long_resp, max_sentences=2)
    print(f"  compressed: {compressed!r}")
    assert len(compressed) < len(long_resp)
    print("  enable_response_compression: PASSED")

    impact = optimizer.get_optimization_impact()
    print(f"  optimization impact: {impact}")

    # ----------------------------------------------------------------- 3
    print("\nTesting FeedbackLoop...")
    feedback = FeedbackLoop()

    # Engineer lacks authority -> rejected.
    r1 = feedback.submit_correction(
        "What is the travel policy for flights over 8 hours?",
        "There is no specific policy for 8+ hour flights.",
        "Employees can book business class for flights over 8 hours with manager approval.",
        "engineer",
    )
    print(f"  engineer correction: {r1}")
    assert r1["accepted"] is False

    # Manager with a more detailed answer -> accepted.
    r2 = feedback.submit_correction(
        "What is the travel policy for flights over 8 hours?",
        "There is no specific policy for 8+ hour flights.",
        "Employees can book business class for flights over 8 hours with manager approval.",
        "manager",
    )
    print(f"  manager correction: {r2}")
    assert r2["accepted"] is True

    # Manager but correction is not more detailed -> rejected.
    r3 = feedback.submit_correction(
        "What is the PTO policy?",
        "Employees accrue paid time off based on tenure and level.",
        "See the handbook.",
        "manager",
    )
    print(f"  weak manager correction: {r3}")
    assert r3["accepted"] is False
    print("  submit_correction: PASSED")

    assert feedback.validate_correction(1) is True
    assert feedback.validate_correction(0) is False
    print("  validate_correction: PASSED")

    metrics = feedback.get_feedback_metrics()
    print(f"  feedback metrics: {metrics}")
    assert metrics["total_corrections"] == 3
    print("  get_feedback_metrics: PASSED")

    print("\nAll tests passed!")
