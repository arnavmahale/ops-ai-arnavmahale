# Week 5: Agentic AI, RAG, and Operational Constraints

## From Models to Agents

A prediction model takes input and returns output: given features, predict demand. An agent is different: given a question, it decides what to do. It may call tools (database lookup, search, calculation), reason about results, call more tools, and eventually return an answer.

This introduces new operational problems:

- **Agency**: The system's behavior is not fully determined by the input. It has choices to make.
- **Partial observability**: The agent doesn't know the answer immediately; it must gather information.
- **Tool errors**: If a tool fails, the agent must detect and recover.
- **Cost unboundedness**: A prediction model costs X tokens; an agent might call tools multiple times and cost 5X tokens.
- **Latency variability**: A prediction takes predictable time; an agent's time varies based on how many tool calls are needed.

These properties require new operational thinking.

## Retrieval-Augmented Generation (RAG) Pattern

RAG is a pattern where an LLM (language model) is augmented with a retrieval system to answer questions about a knowledge base.

Pattern:
1. User asks a question
2. System retrieves relevant documents from a knowledge base (via keyword search, vector similarity, or database query)
3. System feeds the question + retrieved documents to an LLM
4. LLM generates an answer grounded in the documents

Advantages over pure LLM:
- Answers are grounded in specific documents (verifiable, can cite sources)
- The knowledge base can be updated without retraining the LLM
- Works on domain-specific questions that the base LLM wasn't trained on

Challenges:
- Retrieval must find relevant documents; if retrieval fails, the LLM has no grounding
- LLM can hallucinate even with documents (make up facts, misinterpret retrieved content)
- Ranking matters: if 100 documents are retrieved, the LLM must find the relevant ones among noise

[OpenAI's retrieval best practices emphasize that RAG success depends more on retrieval quality than LLM quality.](https://platform.openai.com/docs/guides/retrieval-augmented-generation) A weak LLM with perfect retrieval outperforms a strong LLM with poor retrieval.

## Tool Use and Agentic Reasoning

An agent is equipped with tools: functions it can call. Tool use creates a reasoning loop:

1. Agent observes the question
2. Agent decides which tool to call (or "I already know the answer")
3. Tool executes and returns a result
4. Agent observes the result
5. If more information is needed, go to step 2. Otherwise, return answer.

Example tools:
- `lookup_policy(policy_name)` → returns policy text
- `query_database(sql)` → returns query results
- `search_documents(query)` → returns top-K documents
- `calculate_impact(parameter)` → returns numerical result

[Tool use in LLMs is formalized as "function calling": the LLM learns to output structured tool invocations, which the system executes.](https://platform.openai.com/docs/guides/function-calling) OpenAI's models, Claude, and others support this.

Advantages:
- LLM decides *what* to do, not just what to say
- Answers ground in real data (database results, document content)

Disadvantages:
- LLM might choose the wrong tool
- Tool execution might fail (database error, document not found)
- Reasoning loop can run forever (agent keeps calling tools but doesn't converge)

## Error Handling and Fallbacks in Agentic Systems

Agents must handle tool failures gracefully:

- **Tool not found**: User asks agent to run a tool that doesn't exist. Agent should say "I can't do that" rather than crash.
- **Tool execution fails**: Database query fails, document search returns nothing, API times out. Agent should explain to user and offer an alternative.
- **Tool result is unhelpful**: Tool returns data, but it doesn't answer the question. Agent should try a different tool or explain the limit.
- **Reasoning loop doesn't converge**: Agent keeps calling tools but never produces an answer. System should timeout and return best-effort answer or "I'm stuck."

Graceful degradation: if sophisticated reasoning fails, fall back to simpler behavior (keyword search instead of semantic search, return most relevant document instead of synthesized answer).

## Cost Management in LLM Applications

LLM API costs are per token: input tokens and output tokens. Agentic systems amplify cost:

- Base query: 100 input tokens + 200 output tokens = 300 tokens
- With tool use: 100 input tokens (question) + 200 output tokens (initial response) + 500 tokens retrieved + 100 tokens tool result + 200 output tokens (final answer) = ~1100 tokens

A 3-4x cost increase is common for agentic systems.

Cost management strategies:

**Token counting**: Track every call. Log input/output tokens. Sum daily/monthly costs. Alert if costs exceed budget.

**Caching**: If the same query is asked multiple times, cache the response. Avoid recomputing.

**Model choice**: Use cheaper models where possible (GPT-3.5 cheaper than GPT-4, but less capable). Use stronger models only for hard queries.

**Tool filtering**: Before calling tools, filter to most likely candidates. If many tools exist, prioritize relevant ones.

**Early exit**: If agent's first answer seems confident, return it. Don't force tool use if the LLM can answer directly.

**Batch processing**: If doing many queries, batch them to take advantage of bulk pricing.

[Major LLM platforms (OpenAI, Anthropic, Google) publish cost per million tokens. Monitor your token usage to estimate monthly spend.](https://openai.com/pricing)

Example math:
- 1M input tokens @ $0.50/M = $0.50
- 1M output tokens @ $1.50/M = $1.50
- Total: $2.00 per 1M tokens (gpt-4-turbo)

If your system uses 10B tokens per month: 10,000 * $2.00 = $20,000/month. Scale matters.

## Access Control and Guardrails

Enterprise systems can't let every user access every tool. Guardrails enforce constraints:

**Role-based access control**: User has role (analyst, executive, data-engineer). Tools are tagged with required roles. Agent checks role before calling tool.
- Example: `query_database` requires `data_engineer` role. Analysts can't run arbitrary SQL.

**Cost limits**: User has monthly budget (e.g., $1000). Agent tracks spend. If user exceeds budget, requests are rejected.

**Rate limiting**: User can make at most 10 queries per minute. Protects system from abuse.

**Output filters**: Agent's response is filtered before returning to user. Redact sensitive information.

Implementation: guardrails are checks before tool execution.

```python
def call_tool(tool_name, user_role, args):
    # Check access
    if tool_name in tools_by_role and user_role not in tools_by_role[tool_name]:
        return error("Access denied")
    
    # Check cost (estimated before execution)
    cost_estimate = estimate_cost(tool_name, args)
    if user_spent + cost_estimate > user_budget:
        return error("Cost limit exceeded")
    
    # Execute
    result = execute_tool(tool_name, args)
    return result
```

[Enterprise LLM guardrails are an active research area.](https://arxiv.org/abs/2401.08917) Many systems require human review before high-cost or high-risk operations.

## Evaluation Metrics for Agents

Evaluating agents is harder than evaluating models because the task itself varies:

**Correctness**: Does the agent answer the question correctly? Requires manual review or comparison to ground truth (expensive).

**Completeness**: Does the agent attempt to answer, or does it give up? (Measured by success rate: % of queries that receive substantive answers.)

**Tool usage**: Does the agent use tools appropriately? (Measured by: wrong tool chosen, tool called unnecessarily, right tool not called when needed.)

**Cost-effectiveness**: What does it cost per query? (Measured by average tokens per query, cost per query, cost per correct answer.)

**Latency**: How long does answering take? (p50, p95, p99 latency. Agents with many tool calls are slower.)

**Failure modes**: What happens when the agent fails? (Graceful degradation, clear error messages, or crashes?)

Practical evaluation: run agent on representative queries, track metrics, categorize failures.

## Safety and Hallucination in Agentic Systems

Agents can hallucinate:
- Invent tool names that don't exist
- Misinterpret tool results (tool says X, agent reports Y)
- Generate false citations (cite documents that don't support the claim)

Mitigation:
- **Grounding**: Every claim in the answer must reference a retrieved document or tool result
- **Tool schema**: Formally define tools so the LLM can't make up functionality
- **Verification**: For high-stakes claims, ask the agent to cite the source
- **Human-in-the-loop**: For sensitive questions, require human review

[Anthropic's research on constitutional AI emphasizes that safety is a process, not a feature: monitoring, feedback loops, and human judgment are essential.](https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback)

## Case Study: A RAG System Failing Silently

Scenario: A TechCorp agent answers HR policy questions. Users rely on answers to make decisions (request time off, understanding benefits, etc.).

Failure mode: The retrieval system searches documents by keyword. A query for "parental leave" retrieves the "parental consent" document (same keywords, wrong domain). The LLM reads the wrong document and answers based on it. The answer is coherent but completely wrong. Users follow the wrong advice.

Silent failure: The system didn't crash, produced a response, seemed confident. No error logs.

Prevention: test RAG on known queries, verify retrieval actually returns relevant documents, validate agent answers against ground truth, implement monitoring for low-confidence answers, require humans to spot-check.

## Operational Maturity Levels for Agents

**Level 1 (Toy)**: Agent works on simple test queries. No monitoring, no cost tracking, no access control.

**Level 2 (Prototype)**: Agent works on wider set of queries. Has basic logging. Cost is tracked but not limited.

**Level 3 (Production)**: Agent has monitoring, cost limits, basic access control. Failures are logged. Performance is tracked.

**Level 4 (Enterprise)**: Full observability, role-based access, rate limiting, human review workflows for high-cost queries, performance SLAs, automated rollback for bad models.

Most enterprises deploy agents at Level 2-3. Level 4 requires significant engineering.

## References

[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- Comprehensive overview of RAG architectures and best practices

[OpenAI's Retrieval Best Practices](https://platform.openai.com/docs/guides/retrieval-augmented-generation)
- Practical guidance on retrieval quality and scaling

[Function Calling with LLMs](https://platform.openai.com/docs/guides/function-calling)
- How to enable and use tool calling in LLM APIs

[Guardrails for Large Language Models](https://arxiv.org/abs/2401.08917)
- Research on constraining LLM behavior and enforcing guardrails

[Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback)
- Anthropic's approach to safety, applicable to agent systems

[LLM API Pricing Comparison (2025)](https://openai.com/pricing)
- Current costs for different models and usage patterns

[Agents as OS: Interacting with Real-World Environments through Language Models](https://arxiv.org/abs/2202.04309)
- Foundational work on agentic reasoning loops and tool use

[Building Reliable AI Applications with Guardrails and Monitoring](https://www.infoq.com/articles/reliable-ai-applications-2024/)
- Practical patterns for production agent systems

[Cost Optimization for LLM-Based Applications](https://www.bedrock.ai/articles/llm-cost-optimization)
- Strategies for reducing LLM API costs at scale
