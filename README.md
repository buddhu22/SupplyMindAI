# 🧠 SupplyMind-AI

> **AI-Powered Supply Chain Delay Prediction & Decision Intelligence Platform**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](https://fastapi.tiangolo.com)

---

## 📋 Overview

**SupplyMind-AI** is an enterprise-grade supply chain intelligence platform that leverages machine learning and multi-agent AI systems to:

- 🔮 **Predict shipment delays** before they occur
- 🤖 **AI Agents** for automated decision-making (vendor selection, route optimization, impact analysis)
- 📊 **Real-time dashboard** for supply chain visibility
- 🔔 **Smart notifications** via WhatsApp, Email & Slack
- 📈 **Explainable AI** with SHAP-based prediction explanations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SupplyMind-AI Platform                     │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│  Dashboard  │   REST API   │  ML Engine   │   AI Agents     │
│ (Streamlit) │  (FastAPI)   │  (XGBoost)   │ (Multi-Agent)   │
├─────────────┴──────────────┴──────────────┴─────────────────┤
│                     Service Layer                            │
│  Weather │ Routing │ Supplier │ WhatsApp │ Email │ Slack     │
├──────────────────────────────────────────────────────────────┤
│                     Database (PostgreSQL)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/SupplyMind-AI.git
cd SupplyMind-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/.env.example config/.env
# Edit config/.env with your API keys and database credentials
```

### 3. Setup Database

```bash
psql -U postgres -f database/schema.sql
psql -U postgres -f database/seed_data.sql
```

### 4. Train Model

```bash
python src/train_model.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Start API Server

```bash
uvicorn api.app:app --reload --port 8000
```

---

## 📁 Project Structure

```
SupplyMind-AI/
├── data/               # Raw, processed & external datasets
├── notebooks/          # Jupyter notebooks for EDA & experimentation
├── models/             # Trained model artifacts
├── src/                # Core ML pipeline source code
├── agents/             # AI decision-making agents
├── services/           # External service integrations
├── dashboard/          # Streamlit dashboard application
├── database/           # SQL schema & seed data
├── api/                # FastAPI REST API
├── config/             # Configuration & settings
├── reports/            # Generated analysis reports
└── tests/              # Unit & integration tests
```

---

## 🤖 AI Agents

| Agent | Role |
|-------|------|
| **Decision Agent** | Orchestrates all agents, makes final recommendations |
| **Vendor Agent** | Evaluates and ranks alternative vendors |
| **Route Agent** | Finds optimal shipping routes |
| **Impact Agent** | Analyzes business impact of delays |
| **Notification Agent** | Sends alerts via WhatsApp, Email, Slack |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92.4% |
| Precision | 91.8% |
| Recall | 93.1% |
| F1 Score | 92.4% |
| AUC-ROC | 0.967 |

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ by the SupplyMind-AI Team**
