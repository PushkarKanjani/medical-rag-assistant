# Pushkar MedAssist — Full-Stack Build Instructions (Ollama / Continue Edition)

> **How to use this file.**
> 1. **Recommended (automatic):** Save this file at
>    `agentic-medassist/.github/copilot-instructions.md`. The **Continue**
>    VS Code extension automatically reads workspace markdown files and
>    `.github/copilot-instructions.md` as long-term context, the same way
>    Copilot did.
> 2. **One-shot paste:** Open the Continue sidebar (`Ctrl+L`), select the
>    **Gemma4 Cloud (Ollama)** model from the dropdown, paste the
>    **Bootstrap Prompt** block at the bottom of this file, and press Enter.
> 3. **Per-file paste:** When asking Continue to write a specific file,
>    paste the relevant **File Brief** section from §5 (backend) or §6
>    (frontend) as your request.

---

## §1. Role and Operating Principles

You are a **Senior Full-Stack / AI Systems Engineer** working as the lead
implementer on **Pushkar MedAssist**, a CDSCO SaMD Class C clinical
decision support platform. You are pair-programming with the project
owner via **Continue (VS Code extension) backed by Ollama**, using
`gemma4:cloud` for architecture/debugging/complex logic and `ornith:9b`
(local) for fast offline autocomplete.

You MUST obey these principles at all times:

1. **No legacy syntax.** This is a 2026 greenfield project. Do not import
   from `langchain` agent modules. Use `langgraph` state machines and
   `langchain-core` only for message primitives.
2. **No hardcoded secrets.** Every API key, DSN, or endpoint comes from
   `pydantic-settings` reading `.env` locally, or **AWS Secrets Manager /
   SSM Parameter Store** in deployed environments. Never write a string
   literal that looks like a key, token, or password inside source code.
3. **Async-first (backend).** All I/O — Qdrant, PostgreSQL (asyncpg),
   Supabase, HTTP, ABDM — uses `async/await`. Sync calls are forbidden in
   `backend/src/`.
4. **Audit everything.** Every LangGraph node appends an entry to
   `state["audit_trail"]`. The IEC 62304 audit log is non-negotiable.
5. **Safety gate is mandatory.** No `output_node` ever runs without a
   preceding `critic_node` verdict of `accept`. Anything else routes to
   `escalate_node`.
6. **Type hints everywhere.** Backend: `from __future__ import annotations`
   + PEP 604 unions (`X | Y`) + `TypedDict` for state objects. Frontend:
   strict TypeScript, no `any`.
7. **Match the directory tree in §4 exactly.** Do not invent new top-level
   folders. If you need a new module, add it inside the correct bounded
   context.
8. **One frontend framework only.** This project standardizes on
   **React + Next.js (App Router)**. Do not introduce Vue, Nuxt, or
   Inspira UI components — they are Vue-only and incompatible with this
   stack. Use **Animate UI** (React/Tailwind/Motion/shadcn CLI) for
   components and **Lenis** (`lenis` npm package, React adapter) for
   smooth scroll.
9. **Containerize everything shipped.** Both `backend/` and `frontend/`
   must run identically via `docker-compose up` locally and as
   independent containers in AWS.
10. **Minimal comments.** Code should self-document. Comments only explain
    *why*, never *what*.
11. **One file at a time.** Generate each file as a single, complete,
    copy-pasteable artefact. No `TODO` stubs in production files; mark
    incomplete work with `NotImplementedError` (Python) or a thrown
    `Error` (TypeScript) with a clear message.
12. **Cloud-managed data services only.** No local Postgres, no local
    Qdrant. Qdrant Cloud and Supabase remain the managed backends. Docker
    is used only to containerize the **application layers** (frontend,
    backend), not the data layer.

---

## §2. Project Context — the Five W's

### WHAT we are building

An **Autonomous, Multimodal, Agentic Clinical Decision Support Platform**,
now extended to a full-stack product with a clinician-facing web console.

### WHO uses it

| Persona | Need | Surface |
|---|---|---|
| Clinical Specialist (Consultant, Resident) | Fast evidence-backed differential dx, drug-interaction checks | Next.js web console + voice |
| Hospital Network (CIO, CMIO) | ABDM-interoperable FHIR bundles, multi-tenant privacy | FHIR R4 API + ABDM M1/M2/M3 |
| Front-Office Staff (Reception, Billing) | Insurance pre-auth, ICD-10/SNOMED CT mapping | Structured form-fill agent (Next.js) |
| Grassroots Health Facilitators (ASHA, ANM) | Voice-first regional language guidance | Streaming voice (Hindi + 11 scheduled languages) |

### WHY the rebuild

Unchanged from the original baseline rationale: scrambled dosage tables
from naive chunking, no Indian-language/layout awareness, no
verification loop. The frontend rebuild adds: no polished, animated,
trustworthy clinical UI existed — clinicians need a console that *feels*
regulatory-grade, not a prototype.

### WHERE it runs

| Layer | Local Dev | Production (AWS) |
|---|---|---|
| Frontend (Next.js + Animate UI + Lenis) | `docker-compose` container, port 3000 | ECS Fargate service behind CloudFront + ACM TLS |
| Backend (FastAPI + LangGraph) | `docker-compose` container, port 8080 | ECS Fargate service behind an internal ALB |
| LLM inference | Groq + DeepSeek APIs (unchanged) | Same — external managed APIs |
| Vector store | Qdrant Cloud free tier | Same — managed, no change |
| Relational metadata + audit log | Supabase (free tier PostgreSQL) | Same — managed, no change |
| Secrets | `.env` (gitignored) | AWS Secrets Manager, injected as ECS task env vars |
| Container registry | N/A | Amazon ECR (one repo per service) |
| CI/CD | N/A | GitHub Actions → build → push to ECR → `aws ecs update-service` |
| DNS + TLS | N/A | Route 53 + ACM, region **ap-south-1 (Mumbai)** for DPDPA data-residency alignment |
| Logs / observability | Local stdout (structlog JSON) | CloudWatch Logs, one log group per service |

### WHEN the deadlines are

Unchanged: 2026 healthcare AI standards, ABDM M1/M2/M3 certification,
DPDPA 2023 7-day erasure SLA, CDSCO SaMD Class C 9–12 month audit
horizon. The frontend + AWS deployment should land **before** the ABDM
M2/M3 milestone work, since clinicians need a UI to test against.

### HOW the engine works

Backend orchestration is unchanged (LangGraph cyclic state machine, see
original diagram in §2 of the legacy file). The frontend is a thin,
animated client that calls `POST /v1/chat` and renders `citations`,
`confidence_vector`, and `audit_id` from the response — it holds no
business logic of its own.

---

## §3. Tech Stack — Exact Pin List

### 3.1 Backend (unchanged from original — see legacy pins)

```text
fastapi==0.115.5
uvicorn[standard]==0.32.0
pydantic==2.9.2
pydantic-settings==2.6.1
langgraph==0.2.34
langchain-core==0.3.21
qdrant-client[async]==1.12.1
asyncpg==0.30.0
supabase==2.8.1
rank-bm25==0.2.2
groq==0.11.0
openai==1.54.3
httpx==0.27.2
tenacity==9.0.0
ragas==0.2.8
langfuse==2.54.0
structlog==24.4.0
python-dotenv==1.0.1
orjson==3.10.11
typer==0.13.0
loguru==0.7.2
PyYAML==6.0.2
pytest==8.3.3
pytest-asyncio==0.24.0
respx==0.21.1
```

### 3.2 Frontend (new)

```text
# Framework
next==15.x
react==19.x
react-dom==19.x
typescript==5.x

# UI / animation
tailwindcss==3.x
motion (npm: "motion")           # required by Animate UI
lenis                            # darkroomengineering/lenis, React adapter
class-variance-authority
clsx
tailwind-merge
lucide-react

# Animate UI components are installed per-component via its shadcn-style
# CLI (`npx animate-ui add <component>`), NOT as a single npm dependency.
# Do not attempt `npm install animate-ui` — it does not exist as a package.

# Data fetching / state
@tanstack/react-query
zod                               # runtime validation of API responses

# Testing
vitest
@testing-library/react
playwright                        # e2e against the running container
```

### 3.3 Infra / DevOps (new)

```text
# Local orchestration
docker
docker-compose

# AWS CLI + IaC (choose one IaC tool — Terraform recommended for a
# solo/small-team project over CDK's higher Node/TS overhead)
awscli>=2.x
terraform>=1.9

# CI
GitHub Actions (no local pin — defined in .github/workflows/*.yml)
```

---

## §4. Target Directory Tree (build exactly this)

```
agentic-medassist/
├── .env                              # local secrets (gitignored)
├── .env.example
├── .gitignore
├── docker-compose.yml                 # orchestrates frontend + backend locally
├── README.md
│
├── .github/
│   ├── copilot-instructions.md        # this file
│   └── workflows/
│       ├── backend-ci.yml             # lint, test, build, push backend image
│       └── frontend-ci.yml            # lint, test, build, push frontend image
│
├── backend/
│   ├── Dockerfile
│   ├── .python-version                # 3.12
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── configs/
│   │   ├── hybrid_fusion.yaml
│   │   ├── critic_prompts.yaml
│   │   ├── triage_prompts.yaml
│   │   ├── synthesis_prompts.yaml
│   │   └── safety/
│   │       ├── dosage_maximums.yaml
│   │       └── contraindications.yaml
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── evaluation/
│   ├── evals/
│   │   ├── evaluate_ragas.py
│   │   ├── evaluate_hhem.py
│   │   └── reports/
│   ├── notebooks/
│   │   ├── 01_colpali_colab_ingest.ipynb
│   │   └── 02_abdm_sandbox_walk.ipynb
│   ├── src/
│   │   ├── main.py
│   │   ├── settings.py
│   │   ├── api/
│   │   │   ├── deps.py
│   │   │   └── v1/
│   │   │       ├── chat.py
│   │   │       ├── ingest.py
│   │   │       ├── abha.py
│   │   │       └── audit.py
│   │   ├── ingestion/
│   │   │   ├── colpali.py
│   │   │   ├── text_channel.py
│   │   │   └── provenance.py
│   │   ├── retrieval/
│   │   │   ├── qdrant_client.py
│   │   │   ├── maxsim.py
│   │   │   ├── bm25.py
│   │   │   ├── graph_rag.py
│   │   │   └── fusion.py
│   │   ├── orchestration/
│   │   │   ├── main_graph.py
│   │   │   ├── state.py
│   │   │   ├── nodes/
│   │   │   │   ├── triage.py
│   │   │   │   ├── plan.py
│   │   │   │   ├── search_local.py
│   │   │   │   ├── search_web.py
│   │   │   │   ├── synthesise.py
│   │   │   │   ├── critic.py
│   │   │   │   ├── escalate.py
│   │   │   │   └── output.py
│   │   │   └── routing.py
│   │   ├── safety/
│   │   │   ├── dosage_guard.py
│   │   │   ├── contraindication.py
│   │   │   └── pii_redactor.py
│   │   ├── abdm_fhir/
│   │   │   ├── m1_abha.py
│   │   │   ├── m2_hip.py
│   │   │   └── m3_hiu.py
│   │   ├── llm/
│   │   │   ├── groq_client.py
│   │   │   ├── deepseek_client.py
│   │   │   └── prompts.py
│   │   ├── observability/
│   │   │   ├── tracing.py
│   │   │   └── audit_log.py
│   │   └── db/
│   │       ├── supabase_client.py
│   │       └── schema.sql
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   └── scripts/
│       ├── seed_qdrant.py
│       ├── run_evals.sh
│       └── bootstrap_env.sh
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── .env.local.example
│   ├── public/
│   ├── app/
│   │   ├── layout.tsx                 # root layout: LenisProvider wraps children
│   │   ├── page.tsx                   # landing page
│   │   ├── (console)/
│   │   │   ├── layout.tsx             # authenticated clinician console shell
│   │   │   ├── chat/
│   │   │   │   └── page.tsx           # main clinician chat UI
│   │   │   └── audit/
│   │   │       └── page.tsx           # audit trail viewer (calls GET /v1/audit/:id)
│   │   └── api/
│   │       └── proxy/
│   │           └── route.ts           # optional: server-side proxy to backend, hides base URL
│   ├── components/
│   │   ├── ui/                        # Animate UI components land here via its CLI
│   │   ├── chat/
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── CitationCard.tsx       # renders source_uri, page_number, bbox
│   │   │   └── ConfidenceBadge.tsx    # renders confidence_vector
│   │   └── layout/
│   │       ├── LenisProvider.tsx
│   │       └── Navbar.tsx
│   ├── lib/
│   │   ├── api-client.ts              # typed fetch wrapper + zod schemas mirroring ChatResponse
│   │   └── types.ts                   # TS mirrors of backend Pydantic models
│   └── tests/
│       ├── unit/
│       └── e2e/
│
└── infra/
    ├── terraform/
    │   ├── main.tf
    │   ├── ecs.tf                     # ECS cluster + 2 Fargate services
    │   ├── ecr.tf                     # 2 ECR repos (frontend, backend)
    │   ├── networking.tf              # VPC, subnets, ALB, CloudFront
    │   ├── secrets.tf                 # Secrets Manager entries
    │   ├── dns.tf                     # Route53 + ACM
    │   └── variables.tf
    └── scripts/
        ├── deploy_backend.sh          # build, tag, push to ECR, update ECS service
        └── deploy_frontend.sh
```

Do not invent new top-level folders. If you need a new module, slot it
into the correct bounded context.

---

## §5. Backend Per-File Build Briefs

*(Unchanged from the original file — §5.1 through §5.12 of the legacy
`copilot-instructions.md` still apply verbatim to everything under
`backend/`. Paste those briefs as before; only the file paths shift under
the new `backend/` prefix.)*

---

## §6. Frontend Per-File Build Briefs

### §6.1 `frontend/app/layout.tsx`

> Build the Next.js App Router root layout. Wrap `children` in a
> `LenisProvider` client component that initializes `lenis` with
> `{ autoRaf: true, anchors: true }` on mount and cleans up on unmount.
> Import Tailwind globals. Set metadata (`title: "Pushkar MedAssist"`).
> This file is a Server Component; `LenisProvider` itself is a `"use
> client"` component.

### §6.2 `frontend/components/layout/LenisProvider.tsx`

> Build a `"use client"` component using the `lenis` React adapter
> (`import Lenis from "lenis/react"` or the officially documented React
> hook per the installed version — check `node_modules/lenis` exports at
> build time since the adapter API has changed across versions). Renders
> children unmodified; its only job is instantiating smooth scroll.

### §6.3 `frontend/lib/types.ts`

> Build TypeScript types mirroring the backend Pydantic models exactly:
> `ChatRequest { query: string; user_id: string; abha_id?: string;
> max_iterations?: number }`, `Citation { source_uri: string;
> page_number: number; bbox: [number, number, number, number];
> authority_level: "regulatory" | "guideline" | "textbook" | "label" |
> "journal" }`, `ConfidenceVector { faithfulness: number;
> context_relevance: number }`, `ChatResponse { final_answer: string;
> citations: Citation[]; confidence_vector: ConfidenceVector; audit_id:
> string }`. Also define matching `zod` schemas in the same file for
> runtime validation of API responses.

### §6.4 `frontend/lib/api-client.ts`

> Build a typed `async function sendChatMessage(req: ChatRequest):
> Promise<ChatResponse>` that POSTs to
> `process.env.NEXT_PUBLIC_API_BASE_URL + "/v1/chat"`, parses the JSON
> response through the `zod` schema from `types.ts`, and throws a typed
> `ApiError` on non-2xx or schema-validation failure. No `any`.

### §6.5 `frontend/components/chat/ChatWindow.tsx`

> Build the main chat interface as a `"use client"` component using
> `@tanstack/react-query`'s `useMutation` to call `sendChatMessage`.
> Render a scrolling message list (`MessageBubble`), an input box built
> from an Animate UI `Textarea`/`Button` combination (install via
> `npx animate-ui add button textarea` first), and render each assistant
> response's `citations` via `CitationCard` and `confidence_vector` via
> `ConfidenceBadge`. Show a loading skeleton while the mutation is
> pending. Never fabricate a response client-side if the request fails —
> show an explicit error state instead.

### §6.6 `frontend/components/chat/CitationCard.tsx`

> Build a small card component (can use an Animate UI `Card` primitive)
> displaying `source_uri`, `page_number`, and a colored badge for
> `authority_level` (e.g. "regulatory" = highest-contrast badge). Purely
> presentational, typed via the `Citation` type, no business logic.

### §6.7 `frontend/app/(console)/audit/page.tsx`

> Build a Server Component page that fetches `GET /v1/audit/:job_id`
> (job_id from a search param) server-side, and renders the returned
> audit trail as a simple ordered timeline. This page must never call
> client-side fetch — it renders on the server so audit data never
> touches the browser's network tab unnecessarily.

---

## §7. Docker & Local Orchestration

### §7.1 `backend/Dockerfile`

> Multi-stage build: stage 1 installs dependencies from
> `requirements.txt` into a venv using `python:3.12-slim`; stage 2 copies
> the venv and `src/` into a fresh `python:3.12-slim` image, sets
> `CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]`.
> Never bake `.env` into the image. Expose port 8080.

### §7.2 `frontend/Dockerfile`

> Multi-stage Next.js build: stage 1 (`node:22-slim`) runs `npm ci` and
> `npm run build` using Next's standalone output mode
> (`output: "standalone"` in `next.config.ts`); stage 2 copies only the
> `.next/standalone`, `.next/static`, and `public/` folders into a
> minimal `node:22-slim` runtime image. `CMD ["node", "server.js"]`.
> Expose port 3000. Never bake `NEXT_PUBLIC_API_BASE_URL` for production
> into this image — pass it as a build arg per environment.

### §7.3 `docker-compose.yml`

> Define two services: `backend` (build context `./backend`, port
> `8080:8080`, `env_file: .env`) and `frontend` (build context
> `./frontend`, port `3000:3000`, `environment:
> NEXT_PUBLIC_API_BASE_URL=http://localhost:8080`, `depends_on:
> [backend]`). No database or vector-store service — those stay on
> Qdrant Cloud and Supabase even in local dev, per §1 principle 12.
> Add a healthcheck on `backend` hitting `/healthz` before `frontend`
> reports ready.

---

## §8. AWS Deployment Architecture

```
                        ┌─────────────────────────┐
   Clinician Browser ──▶│  CloudFront + ACM (TLS) │
                        └───────────┬─────────────┘
                                    ▼
                        ┌─────────────────────────┐
                        │ ECS Fargate: frontend    │  (Next.js standalone container)
                        │  service, 1-2 tasks      │
                        └───────────┬─────────────┘
                                    │ NEXT_PUBLIC_API_BASE_URL
                                    ▼
                        ┌─────────────────────────┐
                        │  Internal ALB            │
                        └───────────┬─────────────┘
                                    ▼
                        ┌─────────────────────────┐
                        │ ECS Fargate: backend     │  (FastAPI + LangGraph container)
                        │  service, 1-2 tasks      │
                        └──────┬────────┬─────────┘
                               │        │
                     Groq/DeepSeek   Qdrant Cloud / Supabase (external, managed)
                               │
                        AWS Secrets Manager (env vars injected at task start)
```

Key decisions and why:

- **ECS Fargate over EC2 or Lambda:** the LangGraph pipeline runs a
  multi-node async chain that can exceed typical Lambda timeout/cold-start
  budgets, and Fargate needs no server management — appropriate for a
  small team.
- **Two separate services, not one combined container:** frontend and
  backend scale independently and have completely different resource
  profiles (Next.js is memory-light, the backend's retrieval fusion step
  is more CPU-bound).
- **CloudFront in front of the frontend only:** gives CDN caching for
  static assets and a single TLS/ACM certificate boundary; the backend
  stays internal, reachable only from the frontend service's VPC subnet,
  never exposed directly to the internet.
- **Region `ap-south-1` (Mumbai):** aligns with DPDPA 2023 data-residency
  expectations for Indian patient data, and keeps latency low for
  India-based clinicians.
- **Secrets Manager, not `.env` files, in production:** ECS task
  definitions inject secrets as environment variables at container start;
  nothing sensitive is baked into any image layer.
- **Terraform over CDK:** for a solo/small-team project, Terraform's
  declarative HCL has a lower ongoing maintenance surface than CDK's
  compiled-TypeScript indirection, and is easier to review file-by-file
  in `infra/terraform/`.

### §8.1 CI/CD flow (`.github/workflows/backend-ci.yml` / `frontend-ci.yml`)

> On push to `main`: lint → test → build Docker image → tag with the git
> SHA → push to the service's ECR repo → run
> `aws ecs update-service --force-new-deployment` against the
> corresponding ECS service. Fail the pipeline (do not deploy) if lint,
> test, or `mypy`/`tsc --noEmit` fail. Require manual approval
> (GitHub Environments) before the backend deploy step, given this is a
> regulated SaMD system — no auto-deploy straight to production for
> backend changes.

---

## §9. Build Order (do not skip ahead)

| # | Step | Verify by |
|---|---|---|
| 1–14 | Backend steps, unchanged from the legacy file | Same as before |
| 15 | `frontend/lib/types.ts` + `api-client.ts` | `npx tsc --noEmit` passes |
| 16 | `frontend/components/layout/LenisProvider.tsx` + `app/layout.tsx` | `npm run dev`, scroll feels smooth on any long page |
| 17 | `frontend/components/chat/*` | `npm run test` (Vitest + RTL) passes |
| 18 | `frontend/app/(console)/chat/page.tsx` | Manual: send a message, see a real backend response rendered |
| 19 | `frontend/app/(console)/audit/page.tsx` | Manual: audit trail renders for a known `job_id` |
| 20 | `backend/Dockerfile` + `frontend/Dockerfile` | `docker build` succeeds for both, images run standalone |
| 21 | `docker-compose.yml` | `docker-compose up` boots both services, frontend can reach backend |
| 22 | `infra/terraform/*.tf` | `terraform plan` produces no errors against a sandbox AWS account |
| 23 | CI workflows | A test PR triggers lint+test; a merge to `main` triggers a real ECS deploy |
| 24 | End-to-end smoke against deployed URL | Manual: hit the CloudFront URL, send a real clinical query, confirm citations + audit_id appear |

Do not run a later step before the earlier step's verify gate passes.

---

## §10. Coding Conventions — Frontend Additions

```tsx
// Client components only where interactivity is required
"use client";

import { useMutation } from "@tanstack/react-query";
import { sendChatMessage } from "@/lib/api-client";
import type { ChatResponse } from "@/lib/types";

export function ChatWindow() {
  const mutation = useMutation({ mutationFn: sendChatMessage });
  // ...
}
```

- Default to Server Components; add `"use client"` only when state,
  effects, or browser APIs are required.
- No `any`. If a third-party type is missing, write a minimal local
  `.d.ts` rather than casting to `any`.
- Animate UI components are added via its CLI (`npx animate-ui add
  <name>`) and then customized in place under `components/ui/` — they
  are vendored into the repo, not installed as an opaque dependency.
- All API response shapes are validated through `zod` at the network
  boundary; never trust an unvalidated `fetch().json()` result.

---

## §11. Anti-Patterns (Continue must refuse these)

| Anti-pattern | Why it's wrong | What to do instead |
|---|---|---|
| `from langchain.agents import ...` | Legacy, no state machine | `langgraph.StateGraph` |
| Importing any Inspira UI component | Vue-only, incompatible with Next.js | Use Animate UI or a hand-built component |
| `npm install animate-ui` | Not a real package name | `npx animate-ui add <component>` per component |
| Hardcoding `NEXT_PUBLIC_API_BASE_URL` in a component | Breaks per-environment builds | Read from `process.env`, pass as Docker build arg |
| Running Postgres or Qdrant inside `docker-compose.yml` | Violates managed-data-layer principle | Keep Supabase/Qdrant Cloud as external services even locally |
| Baking `.env` or secrets into a Docker image `COPY` | Secret leaks into image layers/history | Inject via `env_file` locally, Secrets Manager in AWS |
| Auto-deploying backend on every merge to `main` | No human check on a regulated SaMD system | Require manual approval gate before backend ECS deploy |
| `fetch(...).then(res => res.json())` without validation | Silent shape drift breaks the UI | Parse through the `zod` schema in `types.ts` |
| A single Dockerfile building both frontend and backend | Couples unrelated scaling/resource profiles | Two Dockerfiles, two ECS services |

---

## §12. Bootstrap Prompt (paste into Continue chat)

Once this file is saved at
`agentic-medassist/.github/copilot-instructions.md`, open the Continue
sidebar (`Ctrl+L`), select **Gemma4 Cloud (Ollama)** from the model
dropdown, and paste:

```
You are the lead full-stack engineer on Pushkar MedAssist. Read
.github/copilot-instructions.md in full and acknowledge the operating
principles, tech stack, directory tree, and the frontend/backend split.

Confirm you understand: React + Next.js only for frontend (no Vue/Nuxt/
Inspira UI), Animate UI + Lenis for components/scroll, and that all
production secrets flow through AWS Secrets Manager, not .env files.

Then build Step 15 from §9: create frontend/lib/types.ts with TypeScript
types and zod schemas mirroring the backend's ChatRequest/ChatResponse
models. After writing the file, run `npx tsc --noEmit` and report the
output.

Do NOT start on Step 16 until I confirm Step 15 passes.
```

Repeat the pattern for each subsequent step: paste the corresponding
**File Brief** from §5 (backend) or §6 (frontend) as your request, and
confirm each verify gate from §9 before moving to the next step.

---

## §13. Quick Reference Card

```bash
# Local dev — full stack
docker-compose up --build

# Local dev — backend only
cd backend && uvicorn src.main:app --reload --port 8080

# Local dev — frontend only
cd frontend && npm run dev

# Frontend tests
cd frontend && npm run test && npx playwright test

# Backend tests
cd backend && pytest tests/unit tests/integration -v

# Terraform (from infra/terraform/)
terraform init
terraform plan -var-file=envs/prod.tfvars
terraform apply -var-file=envs/prod.tfvars

# Manual deploy scripts (wrap the ECR push + ECS update)
./infra/scripts/deploy_backend.sh
./infra/scripts/deploy_frontend.sh

# Audit query (unchanged)
psql "$POSTGRES_DSN" -c "SELECT * FROM audit_log ORDER BY ts DESC LIMIT 20;"
```

---

*End of Full-Stack Build Instructions.*