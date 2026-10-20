# Hugging Face Spaces Deployment Guide

Deploy the **Concrete Strength Predictor** (including the full **TabPFN v2.5** Foundation Model + Classical Champions + Interactive Frontend) to **Hugging Face Spaces** for free.

---

## Why Hugging Face Spaces?
- **16 GB RAM & 2 vCPUs Free**: Easily runs PyTorch and TabPFN without size limits.
- **No Cold Start Timeouts**: Runs as a persistent container.
- **Interactive UI**: Embeds or links your frontend directly.

---

## Step-by-Step Deployment

### Method 1: Connect your GitHub Repo (Recommended & Easiest)

1. **Commit and push all files to your GitHub repository**:
   ```bash
   git add .
   git commit -m "Add Hugging Face Spaces Docker support"
   git push origin main
   ```

2. **Go to Hugging Face Spaces**:
   - Visit [huggingface.co/new-space](https://huggingface.co/new-space).
   - Enter **Space name**: `concrete-strength-predictor`
   - **License**: MIT (or Open Source)
   - **Select the Space SDK**: Choose **Docker** -> **Blank**.
   - **Space hardware**: Free (2 vCPU, 16 GB RAM).
   - Click **Create Space**.

3. **Link to GitHub**:
   - In your newly created Space, go to **Settings** -> **GitHub**.
   - Connect your GitHub repository and branch `main`.
   - Hugging Face will automatically detect the `Dockerfile` and build it.

---

### Method 2: Push Directly via Git to Hugging Face

If you want to push directly from your terminal to Hugging Face without linking GitHub:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   - Choose **Docker** as SDK.
   - Click **Create Space**.

2. Add the Hugging Face Space git remote in PowerShell:
   ```powershell
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/concrete-strength-predictor
   ```

3. Push your repository:
   ```powershell
   git push hf main
   ```
   *(When prompted, enter your Hugging Face username and your Access Token as the password — get token at https://huggingface.co/settings/tokens)*

---

## Verifying Deployment

Once the build finishes (takes ~2 minutes):
- Your space will show **Running**.
- The interactive UI will load directly in the browser at:  
  `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/concrete-strength-predictor`
- Direct API endpoints:
  - `GET /health`
  - `POST /predict`
