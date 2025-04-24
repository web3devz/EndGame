
# 🚀 Endgame Hackathon Project – _"let me know"_

A sleek Web3 application offering **AI-powered crypto portfolio insights** tailored to your personal risk profile. Built with purpose during the **Endgame Hackathon** 🧠⚡

## 🧩 Project Overview

_"let me know"_ empowers crypto investors to make smarter moves by providing **real-time, AI-driven buy/hold/sell opinions**. The app aggregates insights from multiple AI agents, each following a unique trading philosophy, and personalizes recommendations based on your **risk appetite**.


## ✨ Features

- 🔗 **Connect Wallet**  
  Seamlessly connect your crypto wallet with **Coinbase Wallet** or other providers.

- 📊 **Token Analysis**  
  Access deep-dive analytics from multiple AI perspectives.

- 🧠 **Risk Profiling**  
  Select from investor profiles:  
  - 💥 _High Risk / High Reward_  
  - ⚖️ _Balanced & Strategic_  
  - 🛡️ _Safe & Steady_

- 🤖 **AI Trading Agents**  
  A team of specialized agents offers diversified investment strategies:  
  - Bounce Hunter  
  - Crypto Oracle  
  - Momentum Quant Agent  
  - Simple Moving Average (SMA) Agent  
  - Portfolio Manager Agent

- 📈 **Live Token Metrics**  
  View integrated, real-time token data and analytics.


## 🛠 Tech Stack

### 🖥 Frontend
- Next.js 14 + React 18 + TypeScript  
- Tailwind CSS for modern styling  
- Wagmi & Coinbase OnchainKit for wallet integration  
- Recharts for elegant data visuals  
- Zustand for state handling  
- Axios + Moralis SDK for blockchain interactions

### 🧪 Backend
- Python with FastAPI  
- LangChain for orchestrating AI logic  
- SQLite for lightweight DB management  
- Token Metrics API  
- A variety of custom-built AI agents


## 🚀 Getting Started

### ⚙️ Prerequisites
- Node.js v16+  
- Python 3.9+  
- npm, yarn, or pnpm


### 🧬 Backend Setup

```bash
cd apps/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python main.py
# Access API at http://localhost:8000
```


### 🌐 Frontend Setup

```bash
cd apps/frontend
npm install  # or yarn / pnpm
cp .env.example .env
# Edit .env with your frontend API config
npm run dev  # or yarn dev / pnpm dev
# Open http://localhost:3000
```


## 🧭 API Endpoints

### 🔐 Wallet Endpoints
- `POST /wallets/` – Create or retrieve a wallet  
- `GET /wallets/{address}` – View wallet data  
- `PUT /wallets/risk-profile` – Update risk profile  

### 🤖 Agent Endpoints
- `GET /api/v1/agents/` – Access all trading agents  
- `GET /api/v1/agents/analysis/{token}` – Get token-specific analysis  

### 📈 Token Metrics
- `GET /token-metrics/` – Real-time market metrics
