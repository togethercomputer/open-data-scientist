from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uuid
import os
import shutil
from typing import Any
from session_manager import SessionManager
from code_executor import CodeExecutor

app = FastAPI(title="Python Code Interpreter", version="1.0.0")

session_manager = SessionManager()
code_executor = CodeExecutor()


class CodeRequest(BaseModel):
    code: str
    session_id: str | None = None


class CodeResponse(BaseModel):
    result: Any
    session_id: str
    success: bool
    error: str | None = None


class UploadResponse(BaseModel):
    success: bool
    uploaded_files: list[str]
    error: str | None = None


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload multiple files to the custom_data directory"""
    try:
        uploaded_files = []
        custom_data_path = "/app/custom_data"

        # Clear the custom_data directory to start clean
        if os.path.exists(custom_data_path):
            shutil.rmtree(custom_data_path)

        # Ensure the directory exists
        os.makedirs(custom_data_path, exist_ok=True)

        for file in files:
            if file.filename:
                file_path = os.path.join(custom_data_path, file.filename)

                # Save the uploaded file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                uploaded_files.append(file.filename)

        return UploadResponse(success=True, uploaded_files=uploaded_files)

    except Exception as e:
        return UploadResponse(success=False, uploaded_files=[], error=str(e))


@app.post("/execute", response_model=CodeResponse)
async def execute_code(request: CodeRequest):
    session_id = request.session_id or str(uuid.uuid4())

    try:
        session = session_manager.get_or_create_session(session_id)
        result = await code_executor.execute(request.code, session)

        return CodeResponse(result=result, session_id=session_id, success=True)
    except Exception as e:
        return CodeResponse(
            result=None, session_id=session_id, success=False, error=str(e)
        )


@app.post("/sessions", response_model=dict)
async def create_session():
    session_id = str(uuid.uuid4())
    session_manager.get_or_create_session(session_id)
    return {"session_id": session_id}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "variables": list(session.namespace.keys()),
        "created_at": session.created_at,
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
