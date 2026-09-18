# Remote Incident Summary

- Correlation ID: `e896e1a4-476a-4669-9c21-5674f9d68414`
- Alert ID: `ALRT-0001`
- Service: `staging-api`
- Runbook: `High CPU playbook`

## Executed Plan
1. `retrieve_runbook` with `{'query': 'cpu', 'top_k': 2}`
2. `run_diagnostic` with `{'command': 'kubectl top pod', 'host': 'staging-api'}`
3. `summarize_incident` with `{'alert_id': 'ALRT-0001', 'evidence': ['memory://alerts/latest#ALRT-0001', 'memory://runbooks/index#rb-101', 'diagnostic:{"command":"kubectl top pod","host":"staging-api","result":"All pods healthy; CPU normalized.","step":"step-2"}']}`

## Summary
Incident ALRT-0001: CPU spikes observed on staging-api. Recorded diagnostic result: All pods healthy; CPU normalized. Recommend restart if sustained > 90% for 5 minutes. Capture logs before restart; monitor for recurrence.

## Recommended Actions
- Check pod CPU across nodes
- Capture logs before restart
- Restart service if CPU > 90% for 5 minutes
