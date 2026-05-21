# AIPI 561 — Week 2: Taxi Demand Forecasting API on GKE

**Author:** Arnav Mahale (arnav.mahale@duke.edu)

A FastAPI service that serves a pre-trained LightGBM model predicting NYC taxi
demand by zone and 15-minute window. Containerized with Docker, deployed to
Google Kubernetes Engine, with CI/CD via GitHub Actions.

## Repo layout

```
.
├── .github/workflows/
│   ├── ci.yml           # tests + Docker build (PRs to main, pushes to main/develop)
│   └── cd.yml           # build + push to Artifact Registry + deploy to GKE (main only)
├── week2/
│   ├── backend/         # FastAPI app (main.py, data.py) + small lookup files
│   ├── metadata/        # taxi zone lookups and reference PDFs
│   └── starter/
│       ├── Dockerfile   # multi-stage build, libgomp1 for LightGBM
│       └── k8s/
│           ├── configmap.yaml    # GCS bucket name
│           ├── deployment.yaml   # init container + main container, probes, resources
│           └── service.yaml      # LoadBalancer → external IP
├── design_report.md     # operational decisions (probes, replicas, resources, CI/CD)
├── architecture.md      # Mermaid diagram of GitHub → AR → GKE flow
└── .gitignore           # excludes key.json and other secrets
```

## Endpoints

- `GET /health` — readiness/liveness probe target
- `GET /api/heatmap?hour=&dow=&date=&holiday=` — zone demand for an hour
- `GET /api/forecast?zone_id=&hour=&dow=&date=&steps=` — N-step forecast for a zone
- `GET /api/recommendations?zone_id=&hour=&dow=&date=&n=&holiday=` — best pickup zones

## Running

The Kubernetes manifests under `week2/starter/k8s/` are filled in for the GCP
project `ops-ai-arnavmahale` and the bucket `gs://ops-ai-arnavmahale-data`.
The CD workflow auto-deploys on push to `main`. See `design_report.md` for
operational rationale.
