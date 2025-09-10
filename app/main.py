# File: app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import router API
from app.routes.chat import router as chat_router

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Notion Chatbot",
    description="Backend untuk chatbot dengan UI Webview dan API.",
    version="2.0.0",
)

# --- Middleware CORS ---
origins = ["*"]  # ganti dengan domain frontend di production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware logging (opsional, untuk debug)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"📥 Request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"📤 Response status: {response.status_code}")
    return response

# Static files & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# API router
app.include_router(chat_router)

# Webview UI
@app.get("/web", response_class=HTMLResponse, tags=["Webview UI"])
async def get_webview_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Health check
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "ok", "message": "Welcome!", "docs": "/docs", "ui": "/web"}
