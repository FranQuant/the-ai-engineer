# Incident Handoff Summary

- Terminal state: `escalated`
- Escalation reason: `evidence_inconclusive_after_runbook_check`
- Alert ID: `ALRT-0001`
- Service: `staging-api`
- Symptom: CPU spike on node-3
- Runbook: `High CPU playbook`
- Loops run: 2

## Loops
### loop-1 — correlation ID `09eee67d-b64e-4147-aaeb-276849cba228`
- Outcome: `unresolved` (inconclusive)
- Learn note: Diagnosed staging-api, ran kubectl top pod, outcome inconclusive: node-3 CPU at 88% of limit; no pod restarts; cause not identified.
- Act status: `ok`
- Plan rationale: First pass on ALRT-0001: cpu_spike playbook.
  1. `retrieve_runbook` `{'query': 'cpu', 'top_k': 2}`
  2. `run_diagnostic` `{'command': 'kubectl top pod', 'host': 'staging-api'}`
  3. `summarize_incident` `{'alert_id': 'ALRT-0001'}`

### loop-2 — correlation ID `9f4b0d0b-8ad2-4c0d-946d-da304579e072`
- Outcome: `unresolved` (degraded)
- Learn note: Diagnosed staging-api, ran kubectl logs deploy/staging-api --tail=50, outcome degraded: Repeated GC pauses; sustained CPU 92% on node-3; root cause needs human review.
- Act status: `ok`
- Plan rationale: Prior loop was unresolved (loop-1: Diagnosed staging-api, ran kubectl top pod, outcome inconclusive: node-3 CPU at 88% of limit; no pod restarts; cause not identified.); re-reading the runbook and running one further check.
  1. `retrieve_runbook` `{'query': 'cpu', 'top_k': 2}`
  2. `run_diagnostic` `{'command': 'kubectl logs deploy/staging-api --tail=50', 'host': 'staging-api'}`
  3. `summarize_incident` `{'alert_id': 'ALRT-0001'}`

## Summary
Incident ALRT-0001: CPU spikes observed on staging-api. Recorded diagnostic result: Repeated GC pauses; sustained CPU 92% on node-3; root cause needs human review. Recommend restart if sustained > 90% for 5 minutes. Capture logs before restart; monitor for recurrence.

## Recommended Actions
- Check pod CPU across nodes
- Capture logs before restart
- Restart service if CPU > 90% for 5 minutes

## Escalation
The agent stopped without a healthy diagnostic (evidence_inconclusive_after_runbook_check). A human should review the diagnostics above and decide on a restart.
