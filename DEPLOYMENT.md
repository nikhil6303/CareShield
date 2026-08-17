# Production Deployment Guide: CareShield Member Churn & Retention Advisor

This guide provides step-by-step instructions for deploying the **CareShield Advisor** project for **100% free** using **Render** for the Flask ML REST API backend and **Vercel** for the React SPA frontend.

---

## Deployment Architecture Overview

```
                      Browser / User
                            |
                            v
                  Vercel Production Edge
          https://careshield-advisor.vercel.app
                     (React Frontend)
                            |
                        HTTP / REST
                            |
                            v
               Render.com Production Service
           https://careshield-backend.onrender.com
                     (Flask WSGI API)
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
      Preprocessing     XGBoost ML       SHAP
                            |
                            v
                    Retention Advisor
                            |
                            v
                       JSON Payload
```

---

## STEP 1: Git Repository Setup

Push your repository to GitHub if you have not already done so:

```powershell
# 1. Initialize Git repository
git init

# 2. Stage all files
git add .

# 3. Commit changes
git commit -m "Prepare CareShield project for Vercel and Render deployment"

# 4. Connect your GitHub repository
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/CareShield-Advisor.git
git push -u origin main
```

---

## STEP 2: Deploy Backend to Render.com (100% Free)

1. Sign up / log in to [Render.com](https://render.com).
2. Click **New +** → **Web Service**.
3. Select **Build and deploy from a Git repository**, connect your GitHub account, and select your repository (`CareShield-Advisor`).
4. Configure the Web Service settings:
   - **Name:** `careshield-backend` (or your preferred name)
   - **Language / Environment:** `Python 3`
   - **Region:** Choose the region closest to your users (e.g., Oregon, Ohio, Frankfurt, Singapore)
   - **Branch:** `main`
   - **Root Directory:** Leave blank (root directory)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free`

5. Add Environment Variables on Render:
   - **Key:** `FRONTEND_URL`
   - **Value:** `*` (or your deployed Vercel frontend URL e.g. `https://careshield-advisor.vercel.app`)

6. Click **Create Web Service**.
7. Wait for Render to complete the build and deploy. Once finished, copy your public backend service URL:
   > Example: `https://careshield-backend.onrender.com`

---

## STEP 3: Deploy Frontend to Vercel (100% Free)

1. Sign up / log in to [Vercel.com](https://vercel.com).
2. Click **Add New...** → **Project**.
3. Import your GitHub repository (`CareShield-Advisor`).
4. Configure Project Settings:
   - **Framework Preset:** `Vite`
   - **Root Directory:** Select `frontend` (Click Edit → select `frontend` directory)
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`

5. Expand **Environment Variables**:
   - **Name:** `VITE_API_URL`
   - **Value:** `https://careshield-backend.onrender.com` (Use your actual Render URL from Step 2)

6. Click **Deploy**.
7. Once deployment completes, Vercel will provide your public production URL:
   > Example: `https://careshield-advisor.vercel.app`

---

## STEP 4: Connect Backend CORS & Production Verification

1. Go back to your Render Dashboard for `careshield-backend`.
2. Under **Environment Variables**, update `FRONTEND_URL` to your Vercel frontend URL:
   ```env
   FRONTEND_URL=https://careshield-advisor.vercel.app
   ```
3. Test your live application:
   - Open `https://careshield-advisor.vercel.app` in your web browser.
   - Verify top header shows `Live API Online`.
   - Click **Browse File** or **Load Demo Dataset**.
   - Click **Analyze Dataset** and verify live XGBoost predictions, SHAP risk drivers, and Retention Advisor recommendations.

---

## Environment Variables Reference Table

| Service | Variable Name | Purpose | Example Value |
| :--- | :--- | :--- | :--- |
| **Vercel** (Frontend) | `VITE_API_URL` | Connects React SPA to production Flask REST API | `https://careshield-backend.onrender.com` |
| **Render** (Backend) | `FRONTEND_URL` | Configures Flask CORS allowed origins | `https://careshield-advisor.vercel.app` |
| **Render** (Backend) | `PORT` | Auto-set by Render for WSGI binding | `10000` (Managed automatically by Render) |

---

## Local Production Testing Before Push

To test the production backend locally using Gunicorn:

```powershell
# Install Gunicorn locally
pip install gunicorn

# Run Flask using Gunicorn WSGI server
gunicorn app:app
```
*(Backend runs on `http://127.0.0.1:8000` or port specified by `$env:PORT`)*
