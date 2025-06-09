import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8123"


def test_basic_variable_isolation():
    """Test that basic variables are isolated between sessions."""
    # Create two sessions
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]
    
    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]
    
    # Set different values in each session
    httpx.post(
        f"{BASE_URL}/execute", 
        json={"code": "x = 100\nprint(f'Session 1 x = {x}')", "session_id": session1_id}
    )
    
    httpx.post(
        f"{BASE_URL}/execute", 
        json={"code": "x = 200\nprint(f'Session 2 x = {x}')", "session_id": session2_id}
    )
    
    # Verify each session has its own value
    response1 = httpx.post(
        f"{BASE_URL}/execute", 
        json={"code": "print(f'Session 1 final x = {x}')", "session_id": session1_id}
    )
    assert response1.json()["success"]
    assert "Session 1 final x = 100" in response1.json()["result"]
    
    response2 = httpx.post(
        f"{BASE_URL}/execute", 
        json={"code": "print(f'Session 2 final x = {x}')", "session_id": session2_id}
    )
    assert response2.json()["success"]
    assert "Session 2 final x = 200" in response2.json()["result"]


def test_complex_variable_isolation():
    """Test isolation with complex data structures."""
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]
    
    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]
    
    # Create complex data in session 1
    session1_code = """
data = {
    'numbers': [1, 2, 3, 4, 5],
    'name': 'Session 1 Data',
    'nested': {'inner': 'value1'}
}
result = len(data['numbers'])
print(f'Session 1: {data["name"]}, length = {result}')
"""
    
    # Create different complex data in session 2
    session2_code = """
data = {
    'numbers': [10, 20, 30],
    'name': 'Session 2 Data', 
    'nested': {'inner': 'value2'}
}
result = sum(data['numbers'])
print(f'Session 2: {data["name"]}, sum = {result}')
"""
    
    response1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": session1_code, "session_id": session1_id}
    )
    assert response1.json()["success"]
    assert "Session 1 Data, length = 5" in response1.json()["result"]
    
    response2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": session2_code, "session_id": session2_id}
    )
    assert response2.json()["success"]
    assert "Session 2 Data, sum = 60" in response2.json()["result"]
    
    # Verify data is still isolated
    check1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "print(f'Check 1: {data[\"name\"]}')", "session_id": session1_id}
    )
    assert "Session 1 Data" in check1.json()["result"]
    
    check2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "print(f'Check 2: {data[\"name\"]}')", "session_id": session2_id}
    )
    assert "Session 2 Data" in check2.json()["result"]


def test_function_definition_isolation():
    """Test that function definitions are isolated between sessions."""
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]
    
    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]
    
    # Define different functions in each session
    function1_code = """
def calculate(x):
    return x * 10 + 1

result = calculate(5)
print(f'Session 1 function result: {result}')
"""
    
    function2_code = """
def calculate(x):
    return x * 100 + 2

result = calculate(5)
print(f'Session 2 function result: {result}')
"""
    
    response1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": function1_code, "session_id": session1_id}
    )
    assert response1.json()["success"]
    assert "Session 1 function result: 51" in response1.json()["result"]
    
    response2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": function2_code, "session_id": session2_id}
    )
    assert response2.json()["success"]
    assert "Session 2 function result: 502" in response2.json()["result"]
    
    # Verify functions are still different
    check1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "print(f'Recheck 1: {calculate(3)}')", "session_id": session1_id}
    )
    assert "Recheck 1: 31" in check1.json()["result"]
    
    check2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "print(f'Recheck 2: {calculate(3)}')", "session_id": session2_id}
    )
    assert "Recheck 2: 302" in check2.json()["result"]


def test_class_definition_isolation():
    """Test that class definitions are isolated between sessions."""
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]
    
    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]
    
            # Define different classes in each session
    class1_code = """class Calculator:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def calculate(self, x):
        return x * self.multiplier

calc = Calculator(5)
result = calc.calculate(10)
print(f'Session 1 class result: {result}')"""
    
    class2_code = """class Calculator:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def calculate(self, x):
        return x * self.multiplier + 1000

calc = Calculator(3)
result = calc.calculate(10)
print(f'Session 2 class result: {result}')"""
    
    response1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": class1_code, "session_id": session1_id}
    )
    
    # Debug: Print the actual response if it fails
    if not response1.json()["success"]:
        print(f"Session 1 error: {response1.json()}")
    
    assert response1.json()["success"]
    assert "Session 1 class result: 50" in response1.json()["result"]
    
    response2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": class2_code, "session_id": session2_id}
    )
    
    # Debug: Print the actual response if it fails
    if not response2.json()["success"]:
        print(f"Session 2 error: {response2.json()}")
    
    assert response2.json()["success"]
    assert "Session 2 class result: 1030" in response2.json()["result"]


def test_variable_isolation_with_imports():
    """Test that variables remain isolated even when imports are shared."""
    session1_response = httpx.post(f"{BASE_URL}/sessions")
    session1_id = session1_response.json()["session_id"]
    
    session2_response = httpx.post(f"{BASE_URL}/sessions")
    session2_id = session2_response.json()["session_id"]
    
    # Session 1: Import modules and set variables
    session1_code = """
import json
import os
# Set local variables (should be isolated)
my_data = {'session': 1, 'value': 'unique_to_session_1'}
my_number = 42
json_test = json.dumps({'session': 1})
print(f'Session 1 data: {my_data}, number: {my_number}')
print(f'Session 1 JSON: {json_test}')
"""
    
    response1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": session1_code, "session_id": session1_id}
    )
    assert response1.json()["success"]
    result1 = response1.json()["result"]
    assert "Session 1 data: {'session': 1, 'value': 'unique_to_session_1'}, number: 42" in result1
    assert '{"session": 1}' in result1
    
    # Session 2: Import same modules but set different variables
    session2_code = """
import json
import os
# Set different local variables (should be isolated)
my_data = {'session': 2, 'value': 'unique_to_session_2'}
my_number = 99
json_test = json.dumps({'session': 2})
print(f'Session 2 data: {my_data}, number: {my_number}')
print(f'Session 2 JSON: {json_test}')
"""
    
    response2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": session2_code, "session_id": session2_id}
    )
    assert response2.json()["success"]
    result2 = response2.json()["result"]
    # Variables should be isolated
    assert "Session 2 data: {'session': 2, 'value': 'unique_to_session_2'}, number: 99" in result2
    assert '{"session": 2}' in result2
    
    # Verify session 1 variables are still intact and different
    check1 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "print(f'Session 1 final check: {my_data}, {my_number}')", "session_id": session1_id}
    )
    assert check1.json()["success"]
    assert "Session 1 final check: {'session': 1, 'value': 'unique_to_session_1'}, 42" in check1.json()["result"]
    
    # Verify session 2 variables are still intact and different
    check2 = httpx.post(
        f"{BASE_URL}/execute",
        json={"code": "print(f'Session 2 final check: {my_data}, {my_number}')", "session_id": session2_id}
    )
    assert check2.json()["success"]
    assert "Session 2 final check: {'session': 2, 'value': 'unique_to_session_2'}, 99" in check2.json()["result"]


def test_concurrent_variable_isolation():
    """Test variable isolation under concurrent access."""
    
    def create_session_with_unique_data(session_number: int):
        """Create a session with unique variables."""
        session_response = httpx.post(f"{BASE_URL}/sessions")
        session_id = session_response.json()["session_id"]
        
        # Set unique data for this session
        unique_code = f"""
session_id_number = {session_number}
session_data = {{
    'id': {session_number},
    'name': 'Session_{session_number}',
    'values': [i * {session_number} for i in range(5)]
}}
computed_value = sum(session_data['values'])
print(f'Session {{session_id_number}} computed: {{computed_value}}')
"""
        
        response = httpx.post(
            f"{BASE_URL}/execute",
            json={"code": unique_code, "session_id": session_id}
        )
        
        assert response.json()["success"]
        
        # Verify the computed value is correct
        expected_sum = sum([i * session_number for i in range(5)])  # 0+n+2n+3n+4n = 10n
        assert f"Session {session_number} computed: {expected_sum}" in response.json()["result"]
        
        return session_number, session_id, expected_sum
    
    # Run 10 concurrent sessions
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_session = {
            executor.submit(create_session_with_unique_data, i): i 
            for i in range(1, 11)
        }
        
        results = []
        for future in as_completed(future_to_session):
            session_number = future_to_session[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Session {session_number} failed: {e}")
                raise
    
    # Verify all sessions computed correctly
    assert len(results) == 10
    for session_number, session_id, expected_sum in results:
        assert expected_sum == session_number * 10  # 0+n+2n+3n+4n = 10n
    
    print(f"✅ All {len(results)} concurrent sessions maintained variable isolation") 