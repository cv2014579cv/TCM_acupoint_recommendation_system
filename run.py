import uvicorn
import os
import sys

# 獲取根目錄路徑
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

if __name__ == "__main__":
    print("正在啟動中醫經穴推薦系統後端...")
    print("後端地址: http://localhost:8000")
    print("前端地址: 請在瀏覽器中打開 web/index.html")
    
    # 使用 模組化路徑 "backend.main:app" 以避免路徑混淆
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
