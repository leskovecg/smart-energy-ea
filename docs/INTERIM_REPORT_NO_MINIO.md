# Smart Energy Active Learning – Interim Report (No MinIO)

## 1. Objective
Reduce simulator usage and runtime by applying Active Learning (AL) to N-1 security assessment
while maintaining classification performance.

## 2. Data & Digital Twin
- Dataset: `simulation_security_labels_n-1.csv`
- Labels: secure / insecure
- Digital twin: `digital_twin_ext_grid.json`

## 3. Methods
- Model: Random Forest
- AL strategies: entropy, uncertainty, margin, random
- On-demand simulator labeling with caching

## 4. KPIs
- Total labeled samples
- Sample saving ratio
- Simulator calls (cumulative)
- Simulator runtime (seconds)

## 5. Experiments
- Single-run comparison: entropy vs random
- Offline grid comparison across strategies

## 6. Results
(To be filled after execution)

## 7. Next Steps
- Integrate MinIO for persistent storage
- Bind results to HumAIne dashboard
- Extend simulator scenarios
