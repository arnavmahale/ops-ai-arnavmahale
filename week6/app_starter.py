"""
Week 6: Agent Architecture + Guardrails (TechCorp LLM Agent)

Built on the Week 5 agent. An AI agent that answers TechCorp business questions
by combining:
- Gemini 2.5 Pro LLM (free tier via Google AI API) for reasoning / routing
- SQLite database queries (employees / expenses / projects / benefits)
- Policy document retrieval (documents.json) and expense limits (policies.json)

Reasoning loop (two LLM calls per query):
  1. ROUTE  - LLM reads the system prompt + question and decides which tool to
              call (or answers directly if no tool is needed).
  2. TOOL   - we parse the LLM's "TOOL:/ARGS:" response, run the tool locally,
              and get real data back.
  3. ANSWER - we hand the tool result back to the LLM to synthesize a final,
              human-readable answer.

Tokens + cost are accumulated across BOTH calls so get_metrics() reflects the
true cost of a query.

Week 6 guardrails (see access_control_starter.py):
  - RateLimiter   : blocks a user who exceeds N queries/minute  (BEFORE the LLM)
  - CostEnforcer  : blocks a user who is over their role budget  (BEFORE the LLM)
  - AccessController.redact_response : scrubs sensitive fields the caller's role
                    may not see                                  (AFTER the LLM)
The two pre-LLM guardrails fail fast so a blocked request never spends tokens.
"""

import json
import os
import re
import time
import sqlite3
import logging
from typing import Dict, Any

import google.genai as genai
from google.genai import types

from access_control_starter import AccessController, RateLimiter, CostEnforcer

# Optionally load a local week5/.env (gitignored) so GOOGLE_API_KEY is picked up
# automatically. No-op if python-dotenv isn't installed or there's no .env file.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Resolve data files relative to THIS file so the agent runs from any cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# NOTE: The assignment specifies gemini-2.5-pro, but Google now sets the FREE-tier
# quota for 2.5-pro to zero (it requires a billing-enabled account). gemini-2.5-flash
# is the capable model that IS available on the free tier, so we run on Flash by
# default. Override with the GEMINI_MODEL env var if you enable billing for Pro.
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Cost-tracking rates (USD per 1M tokens). We keep the assignment's documented
# Gemini 2.5 Pro rate card for the cost-tracking exercise; see REPORT.md for the
# note on why execution happens on Flash.
INPUT_RATE = 0.075
OUTPUT_RATE = 0.3


# ---------------------------------------------------------------------------
# TASK 1: Tool base class
# ---------------------------------------------------------------------------


class Tool:
    """Base class for tools the agent can call."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        """Run the tool. Subclasses must override this."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 2: EmployeeLookupTool
# ---------------------------------------------------------------------------


class EmployeeLookupTool(Tool):
    """Look up employee information from the SQLite database."""

    def __init__(self, db_path: str):
        super().__init__("employee_lookup", "Find employee information by name or ID")
        self.db_path = db_path

    def execute(self, employee_name: str = None, employee_id: str = None) -> str:
        """Look up an employee by name (partial match) or id (exact match).

        Returns a JSON string of matching rows, or "Employee not found".
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # lets us return dicts keyed by column
            cursor = conn.cursor()

            if employee_id is not None and str(employee_id).strip():
                cursor.execute(
                    "SELECT * FROM employees WHERE id = ?", (employee_id,)
                )
            elif employee_name is not None and str(employee_name).strip():
                cursor.execute(
                    "SELECT * FROM employees WHERE name LIKE ? LIMIT 10",
                    (f"%{employee_name}%",),
                )
            else:
                conn.close()
                return "Employee not found (no name or id provided)"

            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()

            if not rows:
                return "Employee not found"
            return json.dumps(rows, indent=2, default=str)
        except Exception as e:
            logger.error(f"Employee lookup error: {e}")
            return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# TASK 3: PolicySearchTool
# ---------------------------------------------------------------------------


class PolicySearchTool(Tool):
    """Search policy documents by keyword."""

    def __init__(self):
        super().__init__(
            "policy_search", "Search policy documents by keyword or topic"
        )
        # Load documents once, at construction.
        with open(os.path.join(DATA_DIR, "documents.json")) as f:
            self.documents = json.load(f)

    # very small stopword list so phrases like "the travel policy" still match
    _STOPWORDS = {"the", "a", "an", "of", "for", "is", "what", "and", "to", "on"}

    def execute(self, query: str, limit: int = 5) -> str:
        """Return up to `limit` documents most relevant to `query`.

        We score each document by how many distinct query words appear in its
        title/content (title matches weighted higher), then return the top hits.
        Word-level scoring means a phrase like "travel policy" still finds the
        "Travel and Expense Policy" doc even though that exact phrase never
        appears verbatim. Each hit returns title + a 500-char snippet.
        """
        try:
            if not query:
                return "No query provided"

            words = [
                w
                for w in re.findall(r"[a-z0-9]+", query.lower())
                if w not in self._STOPWORDS
            ]
            if not words:
                words = [query.lower()]

            scored = []
            for doc in self.documents:
                title = doc.get("title", "").lower()
                content = doc.get("content", "").lower()
                score = sum(
                    (2 if w in title else 0) + (1 if w in content else 0)
                    for w in words
                )
                if score > 0:
                    scored.append((score, doc))

            if not scored:
                return f"No policy documents found matching: {query}"

            scored.sort(key=lambda x: x[0], reverse=True)

            out = []
            for _, doc in scored[:limit]:
                snippet = doc.get("content", "")[:500].strip()
                out.append(
                    f"### {doc.get('title', 'Untitled')} "
                    f"(category: {doc.get('category', 'n/a')}, "
                    f"sensitivity: {doc.get('sensitivity', 'n/a')})\n{snippet}"
                )
            return "\n\n".join(out)
        except Exception as e:
            logger.error(f"Policy search error: {e}")
            return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# TASK 4: ExpenseQueryTool
# ---------------------------------------------------------------------------


class ExpenseQueryTool(Tool):
    """Query expense approval limits by role."""

    def __init__(self):
        super().__init__("expense_query", "Query expense approval limits by role")
        with open(os.path.join(DATA_DIR, "policies.json")) as f:
            self.policies = json.load(f)

    def execute(self, role: str) -> str:
        """Return the approval limit for a role, or a not-found message."""
        try:
            limits = self.policies.get("expense", {}).get("approval_limits", {})
            key = (role or "").strip().lower()
            if key in limits:
                return f"Approval limit for {key}: ${limits[key]}"
            return (
                f"Role not found: {role}. "
                f"Valid roles: {', '.join(limits.keys())}"
            )
        except Exception as e:
            logger.error(f"Expense query error: {e}")
            return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# TASK 5: Agent
# ---------------------------------------------------------------------------


class Agent:
    """AI agent that answers questions using Gemini LLM + tools."""

    def __init__(self, db_path: str, api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or GOOGLE_API_KEY

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Get a free key at: "
                "https://aistudio.google.com/app/apikey"
            )

        self.client = genai.Client(api_key=self.api_key)

        self.tools = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search": PolicySearchTool(),
            "expense_query": ExpenseQueryTool(),
        }

        # Week 6 guardrails.
        self.access_controller = AccessController("data/access_control.json")
        self.rate_limiter = RateLimiter(max_queries_per_minute=30)
        self.cost_enforcer = CostEnforcer()

        # Running metrics
        self.token_count = 0
        self.total_cost = 0.0
        self.queries_run = 0
        self.blocked_queries = 0  # queries refused by a guardrail

    # -- LLM call with retry ------------------------------------------------

    def _generate(self, contents: str, config) -> Any:
        """Call Gemini, retrying on transient 429 rate-limit errors.

        The free tier has tight per-minute limits; on a 429 we back off and
        retry a few times before giving up.
        """
        last_err = None
        for attempt in range(4):
            try:
                return self.client.models.generate_content(
                    model=MODEL, contents=contents, config=config
                )
            except Exception as e:
                last_err = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 8 * (attempt + 1)
                    logger.warning(
                        f"Rate limited (429); retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)"
                    )
                    time.sleep(wait)
                    continue
                raise
        raise last_err

    # -- prompt -------------------------------------------------------------

    def _build_system_prompt(self, user_role: str) -> str:
        """Describe the tools to the LLM and define the routing format."""
        tool_lines = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.values()
        )
        return f"""You are a TechCorp assistant. Answer employee questions using the tools below.
User role: {user_role}

Available tools:
{tool_lines}

Tool argument hints:
- employee_lookup: employee_name=<full or partial name>  OR  employee_id=<number>
- policy_search: query=<keyword or topic>
- expense_query: role=<one of: ic1_ic2, ic3, manager, director, vp>

If a tool is needed, respond with EXACTLY this format and nothing else:
TOOL: <tool_name>
ARGS: <key>=<value>

You may pass multiple args as a comma-separated list, e.g. ARGS: employee_name=Brian Yang
If NO tool is needed, just answer the question directly in plain text."""

    # -- main loop ----------------------------------------------------------

    def query(
        self,
        user_query: str,
        user_id: str = "anonymous",
        user_role: str = "engineer",
    ) -> Dict[str, Any]:
        """Answer a question using LLM routing + tool execution + synthesis,
        wrapped by the Week 6 guardrails.

        Args:
            user_query: the question to answer.
            user_id:    caller identity, used for rate limiting + budget tracking.
            user_role:  caller role, used for redaction + role budget.
        """
        logger.info(f"Processing query from {user_id} ({user_role}): {user_query}")

        # --- Guardrail 1: rate limiting (BEFORE the LLM) -------------------
        if not self.rate_limiter.is_allowed(user_id):
            self.blocked_queries += 1
            logger.warning(f"Rate limit exceeded for {user_id}")
            return {
                "answer": None,
                "error": "Rate limit exceeded",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "user_id": user_id,
            }

        # --- Guardrail 2: cost / budget enforcement (BEFORE the LLM) ------
        estimated_cost = 0.01  # conservative per-query estimate
        if not self.cost_enforcer.can_afford_query(
            user_id, estimated_cost, role=user_role
        ):
            self.blocked_queries += 1
            logger.warning(f"Budget exceeded for {user_id} ({user_role})")
            return {
                "answer": None,
                "error": "Budget exceeded",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "user_id": user_id,
                "budget_remaining": self.cost_enforcer.get_budget_remaining(
                    user_id, role=user_role
                ),
            }

        input_tokens = 0
        output_tokens = 0

        try:
            system_prompt = self._build_system_prompt(user_role)

            # --- Call 1: routing -------------------------------------------
            route_resp = self._generate(
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
            )
            i_tok, o_tok = self._tokens_of(route_resp)
            input_tokens += i_tok
            output_tokens += o_tok

            route_text = (route_resp.text or "").strip()
            tool_name, args = self._parse_tool_call(route_text)

            # --- Tool execution --------------------------------------------
            if tool_name and tool_name in self.tools:
                tool_result = self.tools[tool_name].execute(**args)
                logger.info(f"Called tool '{tool_name}' with args {args}")

                # --- Call 2: synthesis -------------------------------------
                synth_prompt = (
                    f"User question: {user_query}\n\n"
                    f"You called the tool '{tool_name}' and got this result:\n"
                    f"{tool_result}\n\n"
                    f"Using ONLY the information above, write a clear, concise "
                    f"answer to the user's question. If the result is an error "
                    f"or says 'not found', explain that plainly."
                )
                synth_resp = self._generate(
                    contents=synth_prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                i_tok, o_tok = self._tokens_of(synth_resp)
                input_tokens += i_tok
                output_tokens += o_tok

                answer = (synth_resp.text or "").strip()
            else:
                # No tool needed (or unrecognized tool) -> use the direct reply.
                answer = route_text or "I wasn't able to produce an answer."

            # --- Guardrail 3: redact sensitive fields (AFTER the LLM) ------
            answer = self.access_controller.redact_response(user_role, answer)

            # --- Metrics ---------------------------------------------------
            cost = self._estimate_query_cost(input_tokens, output_tokens)
            total_tokens = input_tokens + output_tokens
            self.token_count += total_tokens
            self.total_cost += cost
            self.queries_run += 1

            # Charge this query against the user's budget.
            self.cost_enforcer.add_cost(user_id, user_role, cost)

            return {
                "answer": answer,
                "tokens_used": total_tokens,
                "cost": cost,
                "role": user_role,
                "user_id": user_id,
                "budget_remaining": self.cost_enforcer.get_budget_remaining(
                    user_id, role=user_role
                ),
                "rate_remaining": self.rate_limiter.get_remaining_queries(user_id),
            }

        except Exception as e:
            logger.exception("Query failed")
            # Still count the query so metrics reflect the attempt.
            cost = self._estimate_query_cost(input_tokens, output_tokens)
            self.token_count += input_tokens + output_tokens
            self.total_cost += cost
            self.queries_run += 1
            self.cost_enforcer.add_cost(user_id, user_role, cost)
            return {
                "answer": f"Error answering query: {e}",
                "tokens_used": input_tokens + output_tokens,
                "cost": cost,
                "role": user_role,
                "user_id": user_id,
            }

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _tokens_of(response) -> tuple:
        """Pull (input_tokens, output_tokens) from a Gemini response safely."""
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return 0, 0
        in_tok = getattr(um, "prompt_token_count", 0) or 0
        out_tok = getattr(um, "candidates_token_count", 0) or 0
        return in_tok, out_tok

    @staticmethod
    def _parse_tool_call(text: str) -> tuple:
        """Parse a 'TOOL: x / ARGS: k=v, k2=v2' block into (name, {args}).

        Returns (None, {}) if the text is a direct answer (no TOOL: line).
        """
        tool_match = re.search(r"TOOL:\s*([a-zA-Z_]+)", text)
        if not tool_match:
            return None, {}
        tool_name = tool_match.group(1).strip()

        args: Dict[str, str] = {}
        args_match = re.search(r"ARGS:\s*(.+)", text)
        if args_match:
            for pair in args_match.group(1).split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    args[k.strip()] = v.strip()
        return tool_name, args

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD given Gemini 2.5 Pro per-token rates."""
        input_cost = (input_tokens / 1_000_000) * INPUT_RATE
        output_cost = (output_tokens / 1_000_000) * OUTPUT_RATE
        return input_cost + output_cost

    def get_metrics(self) -> Dict[str, Any]:
        """Return running totals."""
        avg = self.total_cost / self.queries_run if self.queries_run else 0.0
        return {
            "total_queries": self.queries_run,
            "total_tokens": self.token_count,
            "total_cost": self.total_cost,
            "avg_cost_per_query": avg,
            "blocked_queries": self.blocked_queries,
        }


# ---------------------------------------------------------------------------
# TASK 6: Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def show(result: Dict[str, Any]):
        """Pretty-print one query result (answer or guardrail block)."""
        if result.get("error"):
            print(f"  BLOCKED -> {result['error']}")
            if "budget_remaining" in result:
                print(f"  Budget remaining: ${result['budget_remaining']:.2f}")
            return
        print(f"  Answer: {result['answer'][:300]}")
        print(f"  Tokens: {result['tokens_used']}  Cost: ${result['cost']:.6f}")
        if "rate_remaining" in result:
            print(
                f"  Rate remaining: {result['rate_remaining']}  "
                f"Budget remaining: ${result['budget_remaining']:.2f}"
            )

    try:
        agent = Agent(os.path.join(DATA_DIR, "techcorp.db"))
        print("Agent initialized successfully")

        # ---- 1. Baseline functional queries (unchanged from Week 5) -------
        print("\n=== 1. Functional queries ===")
        functional = [
            ("What is the travel policy?", "u_eng", "engineer"),
            ("What's the expense approval limit for a manager?", "u_eng", "engineer"),
        ]
        for q, uid, role in functional:
            print(f"\nQuery: {q!r}  (user={uid}, role={role})")
            show(agent.query(q, user_id=uid, user_role=role))

        # ---- 2. Access control: same question, different roles -----------
        # An engineer should get salary/SSN redacted; HR sees the full answer.
        print("\n=== 2. Access control (redaction) ===")
        salary_q = "Look up the employee named Brian Yang. Include salary and SSN."
        for role in ("engineer", "hr"):
            print(f"\nQuery: {salary_q!r}  (role={role})")
            show(agent.query(salary_q, user_id=f"u_{role}", user_role=role))

        # ---- 3. Rate limiting --------------------------------------------
        # Drop the limit low and fire several queries from one user.
        print("\n=== 3. Rate limiting (limit lowered to 3/min) ===")
        agent.rate_limiter = RateLimiter(max_queries_per_minute=3)
        for i in range(5):
            res = agent.query(
                "What is the travel policy?", user_id="u_spam", user_role="engineer"
            )
            status = res.get("error") or "OK"
            print(f"  Attempt {i + 1}: {status}")

        # ---- 4. Cost enforcement -----------------------------------------
        # Pre-load spend so the engineer ($100 budget) is already over budget.
        print("\n=== 4. Cost enforcement (engineer $100 budget) ===")
        agent.cost_enforcer.add_cost("u_broke", "engineer", 100.0)
        print(
            f"  Pre-loaded spend; budget remaining: "
            f"${agent.cost_enforcer.get_budget_remaining('u_broke', role='engineer'):.2f}"
        )
        show(agent.query("What is the travel policy?", user_id="u_broke",
                         user_role="engineer"))

        print(f"\n=== Metrics ===\n{agent.get_metrics()}")

    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Error during test")
        sys.exit(1)
