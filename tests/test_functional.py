import httpx
import io
import os
import tempfile
import requests
from open_data_scientist.codeagent import ReActDataScienceAgent

BASE_URL = "http://localhost:8123"


def test_health_check():
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_session():
    response = httpx.post(f"{BASE_URL}/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0


def test_execute_simple_expression():
    response = httpx.post(f"{BASE_URL}/execute", json={"code": "2 + 2"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert data["result"] == 4
    assert "session_id" in data


def test_execute_with_print():
    response = httpx.post(f"{BASE_URL}/execute", json={"code": "print('hello world')"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert data["result"] == "hello world"


def test_stateful_execution():
    session_response = httpx.post(f"{BASE_URL}/sessions")
    session_id = session_response.json()["session_id"]

    response1 = httpx.post(
        f"{BASE_URL}/execute", json={"code": "x = 10", "session_id": session_id}
    )
    assert response1.status_code == 200
    assert response1.json()["success"]

    response2 = httpx.post(
        f"{BASE_URL}/execute", json={"code": "y = x * 2", "session_id": session_id}
    )
    assert response2.status_code == 200
    assert response2.json()["success"]

    response3 = httpx.post(
        f"{BASE_URL}/execute", json={"code": "x + y", "session_id": session_id}
    )
    assert response3.status_code == 200
    data = response3.json()
    assert data["success"]
    assert data["result"] == 30


def test_multiple_sessions_isolation():
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]

    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]

    httpx.post(
        f"{BASE_URL}/execute", json={"code": "value = 100", "session_id": session1_id}
    )

    httpx.post(
        f"{BASE_URL}/execute", json={"code": "value = 200", "session_id": session2_id}
    )

    response1 = httpx.post(
        f"{BASE_URL}/execute", json={"code": "value", "session_id": session1_id}
    )
    assert response1.json()["result"] == 100

    response2 = httpx.post(
        f"{BASE_URL}/execute", json={"code": "value", "session_id": session2_id}
    )
    assert response2.json()["result"] == 200


def test_get_session_info():
    session_response = httpx.post(f"{BASE_URL}/sessions")
    session_id = session_response.json()["session_id"]

    httpx.post(
        f"{BASE_URL}/execute", json={"code": "test_var = 42", "session_id": session_id}
    )

    response = httpx.get(f"{BASE_URL}/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "test_var" in data["variables"]
    assert "created_at" in data


def test_delete_session():
    session_response = httpx.post(f"{BASE_URL}/sessions")
    session_id = session_response.json()["session_id"]

    delete_response = httpx.delete(f"{BASE_URL}/sessions/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"message": "Session deleted"}

    get_response = httpx.get(f"{BASE_URL}/sessions/{session_id}")
    assert get_response.status_code == 404


def test_execute_syntax_error():
    response = httpx.post(f"{BASE_URL}/execute", json={"code": "2 + + +"})
    assert response.status_code == 200
    data = response.json()
    assert not data["success"]
    assert data["error"] is not None


def test_execute_runtime_error():
    response = httpx.post(f"{BASE_URL}/execute", json={"code": "undefined_variable"})
    assert response.status_code == 200
    data = response.json()
    assert not data["success"]
    assert "not defined" in data["error"]


def test_nonexistent_session_get():
    response = httpx.get(f"{BASE_URL}/sessions/nonexistent-id")
    assert response.status_code == 404


def test_nonexistent_session_delete():
    response = httpx.delete(f"{BASE_URL}/sessions/nonexistent-id")
    assert response.status_code == 404


def test_upload_single_file():
    """Test uploading a single file."""
    # Create a fake file
    file_content = "Hello, this is a test file!"
    file_data = io.BytesIO(file_content.encode())

    files = {"files": ("test_file.txt", file_data, "text/plain")}

    response = httpx.post(f"{BASE_URL}/upload", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["success"]
    assert "test_file.txt" in data["uploaded_files"]
    assert len(data["uploaded_files"]) == 1
    assert data["error"] is None


def test_upload_multiple_files():
    """Test uploading multiple files."""
    # Create multiple fake files
    file1_content = "Content of file 1"
    file2_content = "Content of file 2"

    file1_data = io.BytesIO(file1_content.encode())
    file2_data = io.BytesIO(file2_content.encode())

    files = [
        ("files", ("file1.txt", file1_data, "text/plain")),
        ("files", ("file2.txt", file2_data, "text/plain")),
    ]

    response = httpx.post(f"{BASE_URL}/upload", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["success"]
    assert "file1.txt" in data["uploaded_files"]
    assert "file2.txt" in data["uploaded_files"]
    assert len(data["uploaded_files"]) == 2
    assert data["error"] is None


def test_upload_clears_previous_files():
    """Test that uploading new files clears previously uploaded files."""
    # First upload
    file1_content = "First upload file"
    file1_data = io.BytesIO(file1_content.encode())
    files1 = {"files": ("first_file.txt", file1_data, "text/plain")}

    response1 = httpx.post(f"{BASE_URL}/upload", files=files1)
    assert response1.status_code == 200
    assert response1.json()["success"]

    # Second upload (should clear the first file)
    file2_content = "Second upload file"
    file2_data = io.BytesIO(file2_content.encode())
    files2 = {"files": ("second_file.txt", file2_data, "text/plain")}

    response2 = httpx.post(f"{BASE_URL}/upload", files=files2)
    assert response2.status_code == 200

    data2 = response2.json()
    assert data2["success"]
    assert "second_file.txt" in data2["uploaded_files"]
    assert len(data2["uploaded_files"]) == 1  # Only the new file should be present
    assert "first_file.txt" not in data2["uploaded_files"]


def test_upload_with_code_execution():
    """Test that uploaded files can be accessed by code execution."""
    # Upload a test file
    file_content = "test data for code execution"
    file_data = io.BytesIO(file_content.encode())
    files = {"files": ("test_data.txt", file_data, "text/plain")}

    upload_response = httpx.post(f"{BASE_URL}/upload", files=files)
    assert upload_response.status_code == 200
    assert upload_response.json()["success"]

    # Execute code to read the uploaded file
    code = """
import os
file_path = '/app/custom_data/test_data.txt'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    print(f"File exists and contains: {content}")
else:
    print("File not found")
"""

    execute_response = httpx.post(f"{BASE_URL}/execute", json={"code": code})
    assert execute_response.status_code == 200

    data = execute_response.json()
    assert data["success"]
    assert "test data for code execution" in data["result"]


def test_upload_empty_files_list():
    """Test uploading with empty files list."""
    response = httpx.post(f"{BASE_URL}/upload", files=[])
    assert (
        response.status_code == 422
    )  # FastAPI returns 422 when required files are missing


def test_end_to_end_agent_analysis():
    """Download real data, upload it, and verify agent can find columns."""

    # Download a real CSV file
    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    response = requests.get(url)
    response.raise_for_status()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(response.text)
        temp_file_path = f.name

    try:
        # Upload the file via the endpoint
        with open(temp_file_path, "rb") as f:
            files = {"files": ("titanic.csv", f, "text/csv")}
            upload_response = httpx.post(f"{BASE_URL}/upload", files=files)

        assert upload_response.status_code == 200
        assert upload_response.json()["success"]
        assert "titanic.csv" in upload_response.json()["uploaded_files"]

        # Use the agent to analyze the uploaded file in /app/custom_data
        agent = ReActDataScienceAgent(executor="internal", max_iterations=5)

        task = "Your only task: open the csv in the current directory and tell me the columns of the CSV file."
        result = agent.run(task)

        # Verify the agent found the uploaded file and its columns

        assert "PassengerId" in result

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
