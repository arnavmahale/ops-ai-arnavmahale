"""
Week 6 guardrails — demonstration script for the report.

Prints a clean walkthrough of all four guardrails. Only Section 5 makes a live
Gemini call; everything else is deterministic so the output is always clean.

Run:  python3 screenshot_demo.py
"""

import os
from app_starter import Agent
from access_control_starter import AccessController, RateLimiter, CostEnforcer

LINE = "=" * 64


def main():
    agent = Agent(os.path.join("data", "techcorp.db"))

    # ------------------------------------------------------------------ 1
    print(LINE)
    print("1. ACCESS CONTROL — field redaction by role")
    print(LINE)
    ac = agent.access_controller
    sample = "Brian Yang's salary is $467,621 and their SSN is 115-04-4507."
    print(f"Raw answer : {sample}\n")
    for role in ("engineer", "manager", "hr", "finance"):
        print(f"  {role:<9}-> {ac.redact_response(role, sample)}")

    # ------------------------------------------------------------------ 2
    print("\n" + LINE)
    print("2. ACCESS CONTROL — document filtering by role (before the LLM)")
    print(LINE)
    docs = [
        {"id": "d1", "sensitivity": "Public", "content": "Mission statement"},
        {"id": "d2", "sensitivity": "Internal", "content": "Eng handbook"},
        {"id": "d3", "sensitivity": "Confidential", "content": "Salary ranges"},
        {"id": "d4", "sensitivity": "Restricted", "content": "SSN records"},
    ]
    for role in ("engineer", "manager", "executive"):
        visible = [d["id"] for d in ac.filter_documents(role, docs)]
        print(f"  {role:<10}can see: {visible}")

    # ------------------------------------------------------------------ 3
    print("\n" + LINE)
    print("3. RATE LIMITING — 3 queries per minute per user")
    print(LINE)
    rl = RateLimiter(max_queries_per_minute=3)
    for i in range(5):
        ok = rl.is_allowed("user1")
        print(f"  query {i + 1}: {'ALLOWED' if ok else 'BLOCKED (rate limit)'}")

    # ------------------------------------------------------------------ 4
    print("\n" + LINE)
    print("4. COST ENFORCEMENT — engineer has a $100 monthly budget")
    print(LINE)
    ce = CostEnforcer()
    print(f"  budget remaining: ${ce.get_budget_remaining('eng1', role='engineer'):.2f}")
    print("  spend $95 ...")
    ce.add_cost("eng1", "engineer", 95.0)
    print(f"  budget remaining: ${ce.get_budget_remaining('eng1'):.2f}")
    print(f"  can afford a $4 query?  {ce.can_afford_query('eng1', 4.0)}")
    print(f"  can afford a $10 query? {ce.can_afford_query('eng1', 10.0)}  (blocked)")

    # ------------------------------------------------------------------ 5
    print("\n" + LINE)
    print("5. LIVE AGENT QUERY — guardrails wrapped around a real Gemini call")
    print(LINE)
    res = agent.query("What is the travel policy?", user_id="eng1",
                      user_role="engineer")
    answer = res.get("answer") or res.get("error")
    print(f"  Q: What is the travel policy?  (user=eng1, role=engineer)")
    print(f"  A: {answer[:240]}")
    print(f"  cost: ${res.get('cost', 0):.6f}   "
          f"rate left: {res.get('rate_remaining')}   "
          f"budget left: ${res.get('budget_remaining', 0):.2f}")

    # ------------------------------------------------------------------ 6
    print("\n" + LINE)
    print("6. MONITORING — metrics + audit log")
    print(LINE)
    m = agent.get_metrics()
    print(f"  queries: {m['total_queries']}   tokens: {m['total_tokens']}   "
          f"cost: ${m['total_cost']:.6f}   blocked: {m['blocked_queries']}")
    audit = agent.get_audit_log()
    denied = [e for e in audit if not e["allowed"]]
    print(f"  audit log: {len(audit)} access decisions recorded, "
          f"{len(denied)} denied")


if __name__ == "__main__":
    main()
