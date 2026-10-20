# Vercel Deployment Guide — Concrete Strength Predictor

## Deployment Architecture

```
GitHub Repo
    ├── api/index.py          ← Vercel Python Serverless Function
    ├── public/index.html     ← Static frontend (served by Vercel CDN)
    ├── models/classical/     ← Pre-trained ML models (~1 MB total)
    ├── vercel.json           ← Routing & build configuration
    ├── requirements.txt      ← Full local dev dependencies
    └── requirements-vercel.txt ← Minimal Vercel production dependencies
```

### Route Map
| Request | Handled By |
|---------|-----------|
| GET / | public/index.html (static CDN) |
| POST /api/predict | api/index.py (serverless) |
| GET /api/health | api/index.py (serverless) |

---

## Prerequisites

```bash
npm install -g vercel
vercel login
```

---

## Step 1 - Swap requirements for Vercel

Vercel uses requirements.txt automatically. Use the minimal version:

```powershell
# In PowerShell from project root:
Copy-Item requirements.txt requirements-full.txt
Copy-Item requirements-vercel.txt requirements.txt
```

WHY: Full requirements include PyTorch + TabPFN (~800MB) which exceeds
Vercel's 250MB serverless limit. Vercel version uses XGBoost + sklearn
+ pandas (~50MB total).

---

## Step 2 - Deploy

```bash
vercel --prod
```

Prompts:
- Set up and deploy: Yes
- Which scope: your account
- Link to existing project: No
- Project name: concrete-strength-predictor
- Directory: ./

---

## Step 3 - Verify

Your live URLs:
  https://your-project.vercel.app/             -> Frontend
  https://your-project.vercel.app/api/health   -> Health check
  https://your-project.vercel.app/api/predict  -> Predictions

Test the API:
```bash
curl -X POST https://your-project.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"concrete_type":"Bacterial Concrete","curing_age_days":28,"cement":415,"water":190,"fine_aggregate":680,"coarse_aggregate":1140,"bacterial_concentration_cells_ml":1000000}'
```

---

## Model Differences: Local vs Vercel

| Feature              | Local Dev              | Vercel Production     |
|----------------------|------------------------|-----------------------|
| Compressive model    | TabPFN v2.5 (1.62 MPa) | XGBoost (1.62 MPa)   |
| Flexural model       | Random Forest          | Random Forest         |
| Split Tensile model  | XGBoost                | XGBoost               |
| PyTorch required     | Yes                    | No (too large)        |

Both models were Phase 4 audited with equivalent MAE on the test set.

---

## Continuous Deployment

After first deploy, every git push auto-deploys:

```bash
git add .
git commit -m "Update"
git push origin main
```

---

## Troubleshooting

| Issue                  | Fix                                               |
|------------------------|---------------------------------------------------|
| Function timeout       | Cold start ~3s — first request may be slow        |
| Module not found       | Check requirements.txt                            |
| Model file not found   | Ensure models/classical/*.joblib are in git       |
| CORS error             | CORS set to * in api/index.py — should work       |
