   @echo off
   cd /d "%~dp0"
   call venv\Scripts\activate
   uvicorn app:app --reload --port 8000