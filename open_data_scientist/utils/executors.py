import requests
from typing import Any, Dict, Optional
from together import Client
import os
from typing import Optional
import os
from pathlib import Path
from typing import Dict
import sys

assert os.getenv("TOGETHER_API_KEY"), "TOGETHER_API_KEY environment variable must be set"
client = Client()
code_interpreter = client.code_interpreter



def collect_files(directory) -> list[Dict[str, str]]:
    """
    Collects all files from the specified directory and its subdirectories.

    Args:
        directory: The directory to scan for files

    Returns:
        A list of file dictionaries ready for upload to the code interpreter
    """
    files = []
    path = Path(directory)

    if not path.exists():
        print(f"Directory '{directory}' does not exist, skipping file collection")
        return files

    for file_path in Path(directory).rglob("*"):
        if file_path.is_file() and not any(
            part.startswith(".") for part in file_path.parts
        ):
            try:
                # Handle different file types
                if file_path.suffix.lower() in [".csv", ".txt", ".json", ".py", ".log"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    files.append(
                        {
                            "name": str(file_path.relative_to(directory)),
                            "encoding": "string",
                            "content": content,
                        }
                    )
                elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                    print(f"Not uploading excel files")

            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Could not read file {file_path}: {e}")

    return files


def create_tci_session_with_data(data_dir: Optional[str] = None) -> Optional[str]:
    """
    Create a session with optional data file upload

    Args:
        data_dir: Optional directory containing data files to upload

    Returns:
        Session ID if files were uploaded and session created, None otherwise
    """
    session_id = None

    # Handle file uploads if data directory provided
    if data_dir and os.path.exists(data_dir):
        print(f"📁 Collecting files from {data_dir}...")
        files = collect_files(data_dir)

        if files:
            print(
                f"📤 Found {len(files)} files. Initializing session with uploaded files..."
            )

            # Initialize session with files
            init_result = upload_files_tci(files)

            print(init_result)

            if init_result and "session_id" in init_result:
                session_id = init_result["session_id"]
                print(f"✅ Session initialized with ID: {session_id}")
            else:
                print(
                    "⚠️ Failed to get session ID, continuing without persistent session"
                )
        else:
            print("📂 No valid files found in directory")

    return session_id


def execute_code_factory(type: str):
    """
    Factory function to create an executor function based on the type of code to execute.
    """
    if type == "internal":
        # Check if the internal service is healthy before returning the executor
        base_url = os.getenv("CODE_INTERPRETER_URL", "http://localhost:8123")
        try:
            health_response = requests.get(f"{base_url}/health", timeout=5)
            if health_response.status_code != 200:
                raise requests.exceptions.RequestException("Health check failed")
        except requests.exceptions.RequestException:
            print("No docker container available. Use the option '--executor tci' if you don't want to build the container.")
            sys.exit(1)
        return execute_code_internal
    elif type == "tci":
        return execute_code_tci
    else:
        raise ValueError(f"Unsupported code type: {type}")


def execute_code_internal(
    code: str, session_id: str | None = None, files: list[Dict[str, str]] | None = None
) -> dict[str, Any]:
    """
    Execute Python code on the interpreter service and adapt its output.

    Args:
        code: Python code to execute
        session_id: Optional session ID to maintain state between calls
        files: Optional list of files to upload to the code interpreter
              Each file should be a dict with 'name', 'encoding', and 'content' keys
    Returns:
        A dictionary formatted for use with get_execution_summary, including
        status, outputs, errors, and session_id.
    """

    if files:
        raise ValueError("Files are not supported for internal execution")

    base_url = os.getenv("CODE_INTERPRETER_URL", "http://localhost:8123")
    url = f"{base_url}/execute"
    payload: dict[str, Any] = {"code": code}

    if session_id:
        payload["session_id"] = session_id

    response = requests.post(url, json=payload)
    response.raise_for_status()

    raw_response = response.json()

    execution_summary_input: dict[str, Any] = {}
    outputs_list: list[dict[str, Any]] = []
    errors_list: list[str] = []

    # NOTE: that here i am trying to reconstruct the output of TCI, but it is not perfect.
    # Seems to work for most cases, but not all.

    if raw_response.get("success"):
        execution_summary_input["status"] = "success"
        result_data = raw_response.get("result")

        if result_data is not None:
            if isinstance(result_data, dict) and any(
                k in result_data for k in ["image/png", "text/plain"]
            ):
                outputs_list.append({"type": "display_data", "data": result_data})
            elif isinstance(result_data, str) and result_data.startswith(
                "data:image/png;base64,"
            ):
                try:
                    b64_content = result_data.split(",", 1)[1]
                    outputs_list.append(
                        {"type": "display_data", "data": {"image/png": b64_content}}
                    )
                except IndexError:
                    outputs_list.append({"type": "stdout", "data": str(result_data)})
            else:
                outputs_list.append({"type": "stdout", "data": str(result_data)})
    else:
        execution_summary_input["status"] = "failure"
        error_message = raw_response.get("error")
        if error_message:
            errors_list.append(str(error_message))

    execution_summary_input["outputs"] = outputs_list
    execution_summary_input["errors"] = errors_list
    execution_summary_input["session_id"] = raw_response.get("session_id")

    return execution_summary_input


def upload_file_internal(
    files: list[Dict[str, str]] | str, session_id: str | None = None
) -> dict[str, Any]:
    """
    Upload files to the interpreter service custom_data directory.

    Args:
        files: Either a list of files to upload (each file should be a dict with
              'name', 'encoding', and 'content' keys) OR a directory path string
              to collect and upload all files from
        session_id: Optional session ID (not used for upload but kept for consistency)

    Returns:
        A dictionary with upload results including success status and uploaded files.
    """
    import base64
    import io

    # If files is a string, treat it as a directory path and collect files
    if isinstance(files, str):
        from open_data_scientist.utils.executors import collect_files

        directory_path = files
        files = collect_files(directory_path)

        if not files:
            return {
                "success": False,
                "uploaded_files": [],
                "error": f"No valid files found in directory: {directory_path}",
            }

    base_url = os.getenv("CODE_INTERPRETER_URL", "http://localhost:8123")
    url = f"{base_url}/upload"

    # Prepare files for multipart upload
    files_to_upload = []

    for file_info in files:
        file_name = file_info.get("name", "")
        file_encoding = file_info.get("encoding", "utf-8")
        file_content = file_info.get("content", "")

        if not file_name:
            continue

        # Handle different encodings
        if file_encoding == "base64":
            # Decode base64 content
            try:
                decoded_content = base64.b64decode(file_content)
                file_obj = io.BytesIO(decoded_content)
            except Exception as e:
                return {
                    "success": False,
                    "uploaded_files": [],
                    "error": f"Failed to decode base64 content for {file_name}: {str(e)}",
                }
        elif file_encoding == "string":
            # Handle "string" encoding from collect_files (treat as utf-8)
            file_obj = io.BytesIO(file_content.encode("utf-8"))
        else:
            # Treat as text content with specified encoding
            file_obj = io.BytesIO(file_content.encode(file_encoding))

        files_to_upload.append(
            ("files", (file_name, file_obj, "application/octet-stream"))
        )

    try:
        response = requests.post(url, files=files_to_upload)
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "uploaded_files": [],
            "error": f"Upload request failed: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "uploaded_files": [],
            "error": f"Upload failed: {str(e)}",
        }
    finally:
        # Close file objects
        for _, file_tuple in files_to_upload:
            if len(file_tuple) >= 2:
                file_tuple[1].close()


def execute_code_tci(
    code: str,
    session_id: Optional[str] = None,
):
    """
    Executes Python code using Together Code Interpreter and returns the result.
    Args:
        code: The Python code to execute
        session_id: Optional session ID to maintain state between executions

    Returns:
        The output of the executed code as a JSON
    """
    try:
        additional_args: dict[str, Any] = {"code": code, "language": "python"}

        if session_id:
            additional_args["session_id"] = session_id

        response = code_interpreter.run(**additional_args)

        result = {
            "session_id": response.data.session_id,
            "status": response.data.status,
            "outputs": [],
        }

        for output in response.data.outputs:
            result["outputs"].append({"type": output.type, "data": output.data})

        if response.data.errors:
            result["errors"] = response.data.errors

        return result
    except Exception as e:
        error_result = {"status": "error", "error_message": str(e), "session_id": None}
        return error_result


def upload_files_tci(
    files: list[Dict[str, str]],
    session_id: Optional[str] = None,
):
    """
    Uploads files to Together Code Interpreter session.
    Args:
        files: List of files to upload to the code interpreter
              Each file should be a dict with 'name', 'encoding', and 'content' keys
        session_id: Optional session ID to maintain state between executions

    Returns:
        The result of the file upload as a JSON
    """
    try:
        additional_args: dict[str, Any] = {
            "code": 'print("Uploading files...")',
            "files": files,
            "language": "python",
        }

        if session_id:
            additional_args["session_id"] = session_id

        response = code_interpreter.run(**additional_args)

        result = {
            "session_id": response.data.session_id,
            "status": response.data.status,
            "outputs": [],
        }

        for output in response.data.outputs:
            result["outputs"].append({"type": output.type, "data": output.data})

        if response.data.errors:
            result["errors"] = response.data.errors

        return result
    except Exception as e:
        error_result = {"status": "error", "error_message": str(e), "session_id": None}
        return error_result
