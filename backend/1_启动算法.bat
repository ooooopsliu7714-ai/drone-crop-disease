@echo off
chcp 65001 >nul
title 启动 AI 算法服务
echo 正在启动AI算法服务，请稍候...
cd /d "%~dp0"
cd algo_service
call algo_env\Scripts\activate
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8077
pause