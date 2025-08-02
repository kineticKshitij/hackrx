# 🚀 GitHub Deployment Guide for Bajaj HackRX API

Your repository is now configured for deployment on multiple platforms directly from GitHub!

**Repository:** https://github.com/kineticKshitij/hackrx

## 🎯 Quick Deploy Options

### 1. 🚂 Railway (Recommended - Easiest)

**✅ Why Railway:** Free tier, automatic HTTPS, simple GitHub integration

**Steps:**
1. Visit [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "Deploy from GitHub repo"
4. Select `kineticKshitij/hackrx`
5. Railway automatically detects Python and deploys!

**⚡ One-Click Deploy:**
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/3rCu9m?referralCode=bonus)

---

### 2. 🟣 Heroku (Popular Choice)

**Steps:**
1. Visit [heroku.com](https://heroku.com)
2. Create account and install Heroku CLI
3. Create new app from GitHub:
```bash
# From Heroku Dashboard:
# 1. New → Create new app
# 2. Connect to GitHub
# 3. Search for "hackrx" 
# 4. Enable automatic deploys from main branch
```

**⚡ One-Click Deploy:**
[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/kineticKshitij/hackrx)

---

### 3. 🎨 Render (Modern Platform)

**Steps:**
1. Visit [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect repository: `kineticKshitij/hackrx`
5. Render uses the `render.yaml` config automatically

**⚡ One-Click Deploy:**
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/kineticKshitij/hackrx)

---

### 4. ☁️ Azure (Full Production)

**GitHub Actions Workflow:** Already configured in `.github/workflows/deploy.yml`

**Steps:**
1. Create Azure App Service
2. Download Publish Profile from Azure Portal
3. Add to GitHub Secrets as `AZUREAPPSERVICE_PUBLISHPROFILE`
4. Push to `main` branch → Auto-deploys!

---

### 5. 🐙 Vercel (Serverless)

**Steps:**
1. Visit [vercel.com](https://vercel.com)
2. Import GitHub repository
3. Vercel automatically configures for Python

**⚡ One-Click Deploy:**
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/kineticKshitij/hackrx)

---

## 🔧 Configuration Files Added

| File | Purpose | Platform |
|------|---------|----------|
| `.github/workflows/deploy.yml` | GitHub Actions | Azure |
| `Procfile` | Process definition | Heroku |
| `runtime.txt` | Python version | Heroku |
| `render.yaml` | Service config | Render |
| `railway.md` | Instructions | Railway |

## 🌐 Expected Deployment URLs

After deployment, your API will be available at:

- **Railway:** `https://your-app-name.up.railway.app`
- **Heroku:** `https://your-app-name.herokuapp.com`  
- **Render:** `https://your-app-name.onrender.com`
- **Vercel:** `https://your-app-name.vercel.app`

## 📝 API Endpoints Available

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-domain.com/health

# Main API info
curl https://your-domain.com/

# Query endpoint (GET)
curl "https://your-domain.com/query?q=financial-policies"

# Full document processing (POST)
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer d691ab348b0d57d77e97cb3d989203e9168c6f8a88e91dd37dc80ff0a9b213aa" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is covered?", "How to claim?"]
  }' \
  https://your-domain.com/hackrx/run
```

## 🎯 Recommended Quick Start

**For immediate deployment:**

1. **Go to Railway:** [railway.app](https://railway.app)
2. **Sign up with GitHub**
3. **Deploy:** Select your `hackrx` repository
4. **Done!** Your API will be live in ~2 minutes

## 🔄 Automatic Updates

All platforms support automatic deployment:
- Push to `main` branch → Automatic redeploy
- GitHub webhooks trigger builds
- Zero-downtime deployments

## 🛠️ Troubleshooting

**Build fails?**
- Check if `requirements_simple.txt` is being used
- Ensure `main.py` is the startup file
- Check platform-specific logs

**Want full Azure features?**
- Set environment variables in platform dashboard
- Use `requirements_full.txt` instead
- Configure Azure services first

---

## 🏆 Your API is Ready!

Your Bajaj HackRX API is now configured for deployment on multiple platforms directly from GitHub. Choose any platform above and have your API live in minutes!

**Repository:** https://github.com/kineticKshitij/hackrx
