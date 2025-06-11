import httpx
import io
import os
import tempfile
import requests
import asyncio
import pytest
from open_data_scientist.codeagent import ReActDataScienceAgent
from concurrent.futures import ThreadPoolExecutor, as_completed

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


@pytest.mark.asyncio
async def test_concurrent_sessions_isolation():
    """Test that 50 concurrent math operations across different sessions work correctly."""
    
    async def create_session_and_execute(session_number: int):
        """Create a session and execute a unique math operation."""
        async with httpx.AsyncClient() as client:
            # Create session
            session_response = await client.post(f"{BASE_URL}/sessions")
            assert session_response.status_code == 200
            session_id = session_response.json()["session_id"]
            
            # Set a unique variable for this session
            setup_code = f"base_value = {session_number}"
            setup_response = await client.post(
                f"{BASE_URL}/execute", 
                json={"code": setup_code, "session_id": session_id}
            )
            assert setup_response.status_code == 200
            assert setup_response.json()["success"]
            
            # Execute a math operation that depends on the variable
            math_code = "result = base_value * 100 + base_value * 2"
            math_response = await client.post(
                f"{BASE_URL}/execute",
                json={"code": math_code, "session_id": session_id}
            )
            assert math_response.status_code == 200
            assert math_response.json()["success"]
            
            # Get the final result
            result_response = await client.post(
                f"{BASE_URL}/execute",
                json={"code": "result", "session_id": session_id}
            )
            assert result_response.status_code == 200
            assert result_response.json()["success"]
            
            result = result_response.json()["result"]
            expected = session_number * 100 + session_number * 2  # session_number * 102
            
            return session_number, result, expected, session_id

    # Run 50 concurrent operations
    tasks = [create_session_and_execute(i) for i in range(1, 51)]
    results = await asyncio.gather(*tasks)
    
    # Verify all results are correct
    for session_number, actual_result, expected_result, session_id in results:
        assert actual_result == expected_result, f"Session {session_number} (ID: {session_id}) got {actual_result}, expected {expected_result}"
    
    print(f"Successfully verified {len(results)} concurrent sessions with correct isolation")


def test_concurrent_sessions_isolation_sync():
    """Synchronous wrapper for the async concurrent test."""
    asyncio.run(test_concurrent_sessions_isolation())


def test_concurrent_sessions_isolation_threads():
    """Test concurrent sessions using ThreadPoolExecutor instead of asyncio."""
    
    def create_session_and_execute(session_number: int):
        """Create a session and execute a unique math operation using sync httpx."""
        # Create session
        session_response = httpx.post(f"{BASE_URL}/sessions")
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]
        
        # Set a unique variable for this session
        setup_code = f"base_value = {session_number}"
        setup_response = httpx.post(
            f"{BASE_URL}/execute", 
            json={"code": setup_code, "session_id": session_id}
        )
        assert setup_response.status_code == 200
        assert setup_response.json()["success"]
        
        # Execute a math operation that depends on the variable
        math_code = "result = base_value * 100 + base_value * 2"
        math_response = httpx.post(
            f"{BASE_URL}/execute",
            json={"code": math_code, "session_id": session_id}
        )
        assert math_response.status_code == 200
        assert math_response.json()["success"]
        
        # Get the final result
        result_response = httpx.post(
            f"{BASE_URL}/execute",
            json={"code": "result", "session_id": session_id}
        )
        assert result_response.status_code == 200
        assert result_response.json()["success"]
        
        result = result_response.json()["result"]
        expected = session_number * 100 + session_number * 2  # session_number * 102
        
        return session_number, result, expected, session_id

    # Run 50 concurrent operations using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_session = {
            executor.submit(create_session_and_execute, i): i 
            for i in range(1, 51)
        }
        
        results = []
        # Collect results as they complete
        for future in as_completed(future_to_session):
            session_number = future_to_session[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Session {session_number} generated an exception: {e}")
                raise
    
    # Verify all results are correct
    for session_number, actual_result, expected_result, session_id in results:
        assert actual_result == expected_result, f"Session {session_number} (ID: {session_id}) got {actual_result}, expected {expected_result}"
    
    print(f"Successfully verified {len(results)} concurrent sessions with ThreadPoolExecutor")


def test_session_large_data_isolation():
    """Test that large data structures in one session don't affect others."""
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]
    
    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]
    
    # Create a large data structure in session 1
    large_data_code = """
import numpy as np
large_array = np.random.random((1000, 1000))
large_list = list(range(100000))
memory_hog = {'data': large_array, 'list': large_list}
small_value = 123
print(f"Session 1 setup complete, small_value = {small_value}")
"""
    
    response1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": large_data_code, "session_id": session1_id}
    )
    assert response1.json()["success"]
    assert "small_value = 123" in response1.json()["result"]
    
    # Session 2 should work normally and not be affected by session 1's memory usage
    response2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "simple_var = 456\nprint(f'Session 2 value: {simple_var}')", "session_id": session2_id}
    )
    assert response2.json()["success"]
    assert "Session 2 value: 456" in response2.json()["result"]
    
    # Verify session 1 can still access its data
    response1_check = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "print(f'Session 1 small_value: {small_value}')", "session_id": session1_id}
    )
    assert response1_check.json()["success"]
    assert "Session 1 small_value: 123" in response1_check.json()["result"]


def test_session_file_operations_isolation():
    """Test that file operations in different sessions don't interfere."""
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]
    
    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]
    
    # Session 1 creates a file
    file1_code = """
with open('session1_file.txt', 'w') as f:
    f.write('Session 1 data')
print('Session 1 file created')
"""
    response1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": file1_code, "session_id": session1_id}
    )
    assert response1.json()["success"]
    assert "Session 1 file created" in response1.json()["result"]
    
    # Session 2 creates a different file
    file2_code = """
with open('session2_file.txt', 'w') as f:
    f.write('Session 2 data')
print('Session 2 file created')
"""
    response2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": file2_code, "session_id": session2_id}
    )
    assert response2.json()["success"]
    assert "Session 2 file created" in response2.json()["result"]
    
    # Both sessions should be able to read their own files
    read1_code = """
with open('session1_file.txt', 'r') as f:
    content = f.read()
print(f'Session 1 file content: {content}')
"""
    read1_response = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": read1_code, "session_id": session1_id}
    )
    assert read1_response.json()["success"]
    assert "Session 1 file content: Session 1 data" in read1_response.json()["result"]
    
    read2_code = """
with open('session2_file.txt', 'r') as f:
    content = f.read()
print(f'Session 2 file content: {content}')
"""
    read2_response = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": read2_code, "session_id": session2_id}
    )
    assert read2_response.json()["success"]
    assert "Session 2 file content: Session 2 data" in read2_response.json()["result"]



