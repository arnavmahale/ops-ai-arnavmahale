# Week 5 — Agent Test Transcript

**Model used:** `gemini-2.5-flash-lite`  
**Tools:** employee_lookup, policy_search, expense_query  

---

## Query 1 (role: `engineer`)

**Q:** What is the travel policy?

**Answer:**

All business travel must be pre-approved by a manager and adheres to specific budget limits based on employee level and travel type (domestic or international). Domestic travel has per-trip and annual limits that vary for IC1-IC2, IC3-IC4, IC5+, Managers, and Directors+. International travel requires VP approval and has budget limits that are 50% higher.

_tokens: 1138 · cost: $0.000105_

---

## Query 2 (role: `engineer`)

**Q:** What's the expense approval limit for a manager?

**Answer:**

The expense approval limit for a manager is $5000.

_tokens: 320 · cost: $0.000030_

---

## Query 3 (role: `finance`)

**Q:** What is the expense approval limit for a VP?

**Answer:**

The expense approval limit for a VP is $100,000.

_tokens: 323 · cost: $0.000031_

---

## Query 4 (role: `finance`)

**Q:** How much can a director approve in expenses?

**Answer:**

A director can approve up to $25,000 in expenses.

_tokens: 319 · cost: $0.000030_

---

## Query 5 (role: `hr`)

**Q:** Look up the employee named Brian Yang

**Answer:**

Brian Yang is the VP Engineering (Executive) in the Engineering department.

_tokens: 547 · cost: $0.000047_

---

## Query 6 (role: `hr`)

**Q:** Find employee with ID 1

**Answer:**

The employee with ID 1 is Brian Yang.

_tokens: 540 · cost: $0.000046_

---

## Query 7 (role: `engineer`)

**Q:** What does the policy say about remote work?

**Answer:**

The policy states that all employees are eligible for remote work, except for those in roles requiring a physical presence, such as Office Management, Facilities, and Reception.

There are two remote work arrangements:
*   **Full Remote:** Available to all eligible employees. Requires working core hours of 10 am-3 pm PT and participating in in-person events quarterly.
*   **Hybrid:** Involves working 3 days in the office and 2 days remotely. This is preferred for managers with direct reports, with office days typically being Monday-Wednesday.

_tokens: 1078 · cost: $0.000109_

---

## Query 8 (role: `employee`)

**Q:** Tell me about the parental leave / benefits policy

**Answer:**

I am sorry, but the provided information does not contain details about a parental leave or benefits policy. The available documents discuss Paid Time Off (PTO), Crisis Management, and New Product Launch meetings.

_tokens: 807 · cost: $0.000072_

---

## Query 9 (role: `manager`)

**Q:** What is the expense approval limit for an ic3?

**Answer:**

The expense approval limit for an ic3 is $2000.

_tokens: 323 · cost: $0.000030_

---

## Query 10 (role: `engineer`)

**Q:** What is our security policy for handling data?

**Answer:**

The provided documents do not contain a specific security policy for handling data. However, the "GDPR Compliance Policy" outlines some requirements related to data handling, including:

*   Respecting data subject rights such as the right to access, rectification, erasure, restrict processing, and data portability.
*   TechCorp's obligations to maintain a data inventory, conduct DPIAs, report breaches within 72 hours, appoint a DPO, and document processing.
*   Specific rules for employee data handling, which include processing only for HR/payroll/benefits and requiring consent for cross-border transfers.

_tokens: 1121 · cost: $0.000114_

---

## Final Metrics

- **Total queries:** 10
- **Total tokens:** 6516
- **Total cost:** $0.000615
- **Avg cost / query:** $0.000061
