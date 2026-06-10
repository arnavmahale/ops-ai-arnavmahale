"""
Week 5 deliverable: run 10 test queries through the agent and record a transcript.

Covers all three tools (policy_search, expense_query, employee_lookup) plus the
running cost/token metrics. Writes a Markdown transcript to test_results.md that
the report draws from.

Run from the week5/ folder:
    python3 run_tests.py
"""

import os
import time

from app_starter import Agent, DATA_DIR, MODEL

# (question, user_role) — 10 queries exercising every tool.
QUERIES = [
    ("What is the travel policy?", "engineer"),
    ("What's the expense approval limit for a manager?", "engineer"),
    ("What is the expense approval limit for a VP?", "finance"),
    ("How much can a director approve in expenses?", "finance"),
    ("Look up the employee named Brian Yang", "hr"),
    ("Find employee with ID 1", "hr"),
    ("What does the policy say about remote work?", "engineer"),
    ("Tell me about the parental leave / benefits policy", "employee"),
    ("What is the expense approval limit for an ic3?", "manager"),
    ("What is our security policy for handling data?", "engineer"),
]


def main() -> None:
    agent = Agent(os.path.join(DATA_DIR, "techcorp.db"))
    lines = [
        "# Week 5 — Agent Test Transcript",
        "",
        f"**Model used:** `{MODEL}`  ",
        "**Tools:** employee_lookup, policy_search, expense_query  ",
        "",
        "---",
        "",
    ]

    for i, (q, role) in enumerate(QUERIES, 1):
        print(f"[{i}/10] ({role}) {q}")
        result = agent.query(q, user_role=role)
        print(f"      -> {result['answer'][:100].replace(chr(10), ' ')}")
        print(f"      tokens={result['tokens_used']} cost=${result['cost']:.6f}")

        lines += [
            f"## Query {i} (role: `{role}`)",
            "",
            f"**Q:** {q}",
            "",
            f"**Answer:**",
            "",
            result["answer"],
            "",
            f"_tokens: {result['tokens_used']} · cost: ${result['cost']:.6f}_",
            "",
            "---",
            "",
        ]
        time.sleep(4)  # stay under the free-tier per-minute rate limit

    metrics = agent.get_metrics()
    print("\nFINAL METRICS:", metrics)

    lines += [
        "## Final Metrics",
        "",
        f"- **Total queries:** {metrics['total_queries']}",
        f"- **Total tokens:** {metrics['total_tokens']}",
        f"- **Total cost:** ${metrics['total_cost']:.6f}",
        f"- **Avg cost / query:** ${metrics['avg_cost_per_query']:.6f}",
        "",
    ]

    with open(os.path.join(os.path.dirname(__file__), "test_results.md"), "w") as f:
        f.write("\n".join(lines))
    print("\nWrote test_results.md")


if __name__ == "__main__":
    main()
