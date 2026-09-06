@echo off
chcp 65001 >nul
echo ========================================
echo   NR-VQA 视频质量评估 - Demo启动
echo ========================================
echo.

cd /d "%~dp0.."

echo [1/3] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请安装Python 3.8+
    pause
    exit /b 1
)

echo [2/3] 检查模型文件...
if not exist "runs\baseline\model_123.pt" (
    echo ⚠️ 警告: 未找到模型文件 runs\baseline\model_123.pt
    echo    请将训练好的模型重命名为 model_123.pt 放入该目录
    echo    或修改 demo/backend/config.py 中的模型路径
    echo.
    echo    按任意键继续（使用随机权重演示）...
    pause >nul
)

echo [3/3] 启动服务...
echo.
echo 📡 服务地址: http://localhost:8000
echo 📖 API文档: http://localhost:8000/api/docs
echo.
echo 按 Ctrl+C 停止服务
echo ========================================
echo.

python -m uvicorn demo.backend.main:app --host 0.0.0.0 --port 8000 --reload

pause