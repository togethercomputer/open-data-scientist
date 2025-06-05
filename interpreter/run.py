import uvicorn
import os

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8123"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run("main:app", host=host, port=port, reload=reload)
