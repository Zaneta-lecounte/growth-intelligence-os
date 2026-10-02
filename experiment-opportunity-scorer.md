# Experiment Opportunity Scorer

## Purpose
Prioritize growth opportunities using the GROWTH framework.

## GROWTH dimensions
Score 1–5:

### G — Growth Impact
How meaningful could the business impact be?

### R — Research Evidence
How strong is the customer / qualitative evidence?

### O — Opportunity Size
How much traffic, revenue, pipeline, or strategic value is exposed?

### W — Web / Behavioral Evidence
How strong is the behavioral / quantitative signal?

### T — Technical Effort
How difficult is implementation?  
1 = very low effort  
5 = very high effort

### H — Hypothesis Confidence
How confident are we that the proposed change addresses the underlying problem?

## Formula
`Priority Score = (G × R × O × W × H) / T`

## Guardrails
Do not let the numeric score replace judgment.

Also flag:
- strategic dependency
- measurement readiness
- legal / privacy risk
- brand risk
- sample-size risk
- implementation dependency

## Output

| Opportunity | G | R | O | W | T | H | Score | Key Risk |
|---|---:|---:|---:|---:|---:|---:|---:|---|

### Recommendation
- Run now
- Research first
- Instrument first
- Defer
- Reject

### Rationale
...
