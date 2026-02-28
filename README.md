### BACKEND

cd mistral-risk-copilot/backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt

# set env
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/Mac
# edit .env and paste your MISTRAL_API_KEY

uvicorn main:app --reload --port 8000

### FRONTEND
cd mistral-risk-copilot/frontend
python -m venv .venv \n
.venv\Scripts\activate \n
pip install -r requirements.txt \n

streamlit run app.py
