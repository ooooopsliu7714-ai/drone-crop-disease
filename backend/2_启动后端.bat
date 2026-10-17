@echo off
chcp 65001 >nul
title 启动后端服务
echo 正在启动后端服务，请稍候...
cd /d "%~dp0"
cd backend
call .venv\Scripts\activate
uvicorn main:app --reload --port 8001
pause