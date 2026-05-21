# Week 2 Design Report — Taxi Demand API on GKE

**Author:** Arnav Mahale · AIPI 561 Operationalizing AI · Summer 2026

This report documents the operational decisions behind deploying a pre-trained
LightGBM taxi-demand model as a FastAPI service on Google Kubernetes Engine
with GitHub Actions CI/CD. Each subsection explains *what* was chosen and
*why*, in terms of the production tradeoffs covered in the week-2 reading.

## 1. Containerization

The provided Dockerfile is a two-stage build: a `python:3.11-slim` builder
installs dependencies (LightGBM requires the `libgomp1` OpenMP runtime), then a
slim runtime stage copies only the installed packages and application code.
This keeps the final image small, faster to pull on every node, and reduces
attack surface — no build toolchain ships to production. A `HEALTHCHECK` is
defined so the local `docker run` workflow exercises the same `/health`
endpoint Kubernetes will hit in the cluster.

## 2. Kubernetes Deployment

**Replicas: 2.** The assignment requires handling 10+ concurrent requests
without errors. A single replica creates two failure modes: load capacity (one
FastAPI worker may saturate under load) and availability (any pod restart
causes user-visible downtime). Two replicas behind the LoadBalancer Service
spread load and ensure that during a rolling update or a node disruption at
least one healthy pod is always serving traffic.

**Rolling update with `maxSurge: 1` / `maxUnavailable: 1`.** During a deploy,
Kubernetes can create one extra pod above the desired replica count and tear
down one existing pod, giving a smooth handover without doubling resource
usage. Setting `maxUnavailable: 0` would be safer but would block updates if
node capacity is tight; for a 2-replica deployment on a 2-node cluster, 1/1 is
the cheapest configuration that still keeps at least one pod serving traffic
throughout.

**Resource requests vs limits — `512m / 1Gi` requests, `1000m / 3Gi` limits.**
LightGBM inference is moderately CPU-bound and the model + parquet files in
memory push working set into the hundreds of MB. Setting requests to half a
core and 1 GiB schedules pods reliably on `n1-standard-2` nodes (2 vCPU, 7.5
GiB) — two pods per node, with headroom for the init container and node
overhead. Limits at 1 core / 3 GiB allow short bursts and protect against
LightGBM memory spikes on cold-start data loading without inviting noisy-
neighbor problems. This intentionally leaves average utilization low — the
2025 Kubernetes Benchmark notes 10% CPU / 23% memory averages industry-wide,
and that headroom is exactly the cost of reliability.

**Readiness vs liveness probes, deliberately different.** Both hit
`/health`, but with different timings:

| Probe | initialDelaySeconds | periodSeconds | failureThreshold | Intent |
|---|---|---|---|---|
| Readiness | 20 | 10 | 3 | Quickly remove a slow pod from the LoadBalancer pool while it warms up or hiccups |
| Liveness | 60 | 20 | 5 | Slowly decide a pod is actually dead and needs a restart |

The init container downloads ~85 MB from GCS and the FastAPI app loads a
LightGBM model on boot, so the 60s liveness delay prevents Kubernetes from
killing a still-warming pod (the classic restart-cascade failure mode). The
readiness probe is intentionally tighter (20s/10s) so traffic stops hitting
a pod the moment it's struggling, without forcing a restart.

## 3. Service Exposure

A `LoadBalancer` Service in GKE provisions a regional Google Cloud Load
Balancer with an external IP, mapping public port 80 to the container's
port 8000. NodePort would have required clients to know node IPs and high
port numbers; ClusterIP would have been cluster-internal only. The
LoadBalancer is the right choice for a public demo endpoint.

## 4. Data and Model Loading

A two-step pattern keeps the Docker image immutable and small:

1. **Image-baked, version-controlled assets** — `taxi_zones.geojson` (~600 KB) and
   `zone_hour_avg_fare.parquet` (~42 KB) live in `backend/` and are baked into
   the image so the API is functional even if GCS is briefly unavailable for
   read.
2. **GCS-resolved, swappable assets** — the 74 MB demand parquet and 11 MB
   LightGBM model file are downloaded by an `initContainer` from
   `gs://ops-ai-arnavmahale-data` into an `emptyDir` volume shared with the
   main container. This lets us redeploy a model update by replacing the GCS
   object and bouncing the deployment, with no image rebuild.

The init container authenticates with the `github-actions` service account
key mounted from the `gcs-sa-key` Kubernetes Secret. The key is never
committed (it's in `.gitignore`) and never logged.

## 5. CI/CD Pipeline

**Two workflows, single source of truth.** `ci.yml` runs on every push and PR
— installs deps, runs `pytest`, and on `main` does a sanity Docker build
without pushing. `cd.yml` runs only on push to `main`: authenticates to GCP
via the `GCP_SA_KEY` secret, builds the image, pushes both `:latest` and
`:<git-sha>` tags to Artifact Registry, updates the GKE deployment image, and
waits up to 5 minutes on `kubectl rollout status` so a failed deploy fails the
workflow rather than silently succeeding.

The SHA tag is the immutability story: each deploy is traceable to an exact
commit, and rollback is `kubectl set image deployment/demand-api
demand-api=...:<previous-sha>`. The reading's emphasis on immutable artifacts
(never overwrite a known SHA) is enforced by the GitHub Actions build step,
which tags both `:latest` and `:${{ github.sha }}` on every push.

## 6. Auth and Secrets

Authentication uses a service account key stored as the `GCP_SA_KEY` GitHub
Secret, following the week-2 README literally. The key is generated locally
with `gcloud iam service-accounts keys create`, pasted into GitHub Secrets,
and never committed to the repository (`.gitignore` excludes it). A separate
copy of the same key is loaded into a Kubernetes `gcs-sa-key` Secret so the
init container can `gsutil cp` from the private bucket; a docker-registry
Kubernetes Secret (`artifact-registry-secret`) lets pods pull the image from
Artifact Registry.

**Production trade-off:** the course's GCP guide notes that Workload Identity
Federation is preferred over service account keys for GitHub→GCP auth — WIF
exchanges a short-lived OIDC token for a 1-hour access token, with no long-
lived key ever existing. For a production deployment I would migrate to WIF;
for this assignment the SA-key approach matches the README and grading rubric
exactly, and rotation/revocation is a single command (`gcloud iam
service-accounts keys delete`).

## 7. What This Deployment Doesn't Yet Do

Honest list of operational gaps, framed by the week-2 reading:

- **No HPA / VPA.** The replica count is static. Under sustained traffic above
  one pod's capacity the deployment will hit limits before scaling. Adding HPA
  on CPU is a one-liner but introduces the HPA-VPA conflict the reading
  warns about — I'd add HPA only.
- **No structured deployment health beyond pod readiness.** The reading is
  clear that "rollout succeeded" ≠ "system working." Real production would
  layer a smoke test step in `cd.yml` (curl `/api/forecast` with a known input,
  assert non-empty JSON) before declaring success.
- **No canary or blue-green.** A rolling update with `maxUnavailable: 1` puts
  100% of new traffic on the new version as soon as both pods are replaced.
  For a model serving endpoint the right pattern is a canary at 5–25% with
  automated rollback on a latency/error budget; that's a week 4+ concern.
- **No image scanning or signing.** Artifact Registry supports vulnerability
  scanning and Cosign signatures, neither of which are wired up.

## 8. Cost and Cleanup

Resources running for the duration of grading:

| Resource | Approx hourly cost |
|---|---|
| GKE control plane | $0.10/hr |
| 2 × n1-standard-2 nodes | 2 × $0.033 = $0.066/hr |
| LoadBalancer (regional) | ~$0.025/hr |
| **Total** | **~$0.19/hr** |

Sustained for one week that's ~$32, more than half of the $50 student credit.
The cluster is deleted immediately after grading screenshots are captured.
The GCS bucket is retained for Week 3 (per the README); the Artifact Registry
repo is deleted (cheap to recreate later).
