# AIPI 561 — Operationalizing AI

Coursework for Duke AIPI 561 (Summer 2026). Each week extends the previous one;
the same GitHub repo holds all weeks so deployment infrastructure (workflows,
GCP project, GCS bucket) carries forward.

**Author:** Arnav Mahale (`arnav.mahale@duke.edu`)

## Layout

```
.
├── .github/workflows/
│   ├── ci.yml                ← week 2: test + Docker build on push/PR
│   ├── cd.yml                ← week 2: build + push to Artifact Registry + GKE deploy
│   └── validate-data.yml     ← week 3: scheduled data-quality check (hourly cron)
├── week2/                    ← Deployment + CI/CD (taxi demand API on GKE)
│   ├── README.md
│   ├── design_report.md      ← 1-page operational decisions report
│   ├── architecture.md       ← Mermaid diagram of GitHub → AR → GKE flow
│   ├── backend/              ← FastAPI app
│   ├── metadata/             ← taxi zone lookups
│   ├── starter/
│   │   ├── Dockerfile        ← multi-stage, libgomp1 for LightGBM
│   │   └── k8s/              ← Deployment, Service, ConfigMap
│   └── submission/           ← exact files uploaded to Canvas
└── week3/                    ← Data quality validation + graceful degradation
    ├── REPORT.md             ← 3-page report on issues, schedule, degradation
    ├── REPORT.pdf
    ├── backend/data.py       ← extended with check_and_log_data_quality() + cleanup
    └── validation/           ← DataQualityValidator + pytest suite (12 tests)
```

## Per-week summaries

### Week 2 — Deploy taxi demand API to GKE
LightGBM model wrapped in FastAPI, multi-stage Dockerfile, deployed to a 2-node
GKE cluster (`operationalizing-ai`, `us-central1-a`) behind a LoadBalancer.
GitHub Actions handles build, push to Artifact Registry, and rolling-update
deploy on every push to `main`. Cluster + Artifact Registry repo were deleted
after grading; the GCS bucket `gs://ops-ai-arnavmahale-data` is retained.

### Week 3 — Data quality validation
Detects four issue classes in the corrupted upstream parquet
(duplicates, out-of-range `trip_count`, variance collapse on
`cbd_pricing_active`, rate drift on `is_holiday`). Validation runs in three
layers — hourly GitHub Actions, API-startup logging, and inline cleanup on
load — all of them transparent (logged) but never crashing the API. See
`week3/REPORT.md`.

## GCP project

| Resource | Identifier |
|---|---|
| Project | `ops-ai-arnavmahale` |
| GCS bucket | `gs://ops-ai-arnavmahale-data` |
| Artifact Registry | `us-central1-docker.pkg.dev/ops-ai-arnavmahale/docker-repo` (deleted after week 2) |
| GitHub Secret | `GCP_SA_KEY` — service-account JSON used by all workflows |
