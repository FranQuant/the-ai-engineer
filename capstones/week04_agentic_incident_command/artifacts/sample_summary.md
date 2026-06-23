# Remote Incident Summary

- Correlation ID: `bcfa71fd-b3c9-4e12-a5a4-4c6cff0928a6`
- Alert ID: `ALRT-0001`
- Service: `staging-api`
- Runbook: `High CPU playbook`

## Executed Plan
1. `retrieve_runbook` with `{'query': 'cpu', 'top_k': 2}`
2. `run_diagnostic` with `{'command': 'kubectl top pod', 'host': 'staging-api'}`
3. `summarize_incident` with `{'alert_id': 'ALRT-0001', 'evidence': ['retrieve_runbook', 'run_diagnostic', 'summarize_incident']}`

## Summary
Incident ALRT-0001: CPU spikes observed on staging-api. Diagnostics show pods healthy and CPU normalized. Recommend restart if sustained > 90% for 5 minutes. Capture logs before restart; monitor for recurrence.

## Recommended Actions
- Check pod CPU across nodes
- Capture logs before restart
- Restart service if CPU > 90% for 5 minutes
