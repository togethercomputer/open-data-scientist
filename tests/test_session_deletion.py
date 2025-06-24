import pytest
from unittest.mock import patch, MagicMock
from open_data_scientist.codeagent import ReActDataScienceAgent
from open_data_scientist.utils.executors import delete_session_internal


def test_session_cleanup_on_agent_deletion():
    """Test that session is automatically cleaned up when agent object is deleted."""
    
    # Here i need to mock the llm response so at least for this time we do not need to call 
    # any llm (this """"""should""""" be fine as we have other tests for that)
    mock_response = """<think>
I need to execute some Python code to test the session creation.
</think>
<code>
```python
print('hello world')
x = 42
print(f'x = {x}')
```
</code>"""
    
    # Create a mock response object
    mock_choice = MagicMock()
    mock_choice.message.content = mock_response
    
    mock_response_obj = MagicMock()
    mock_response_obj.choices = [mock_choice]
    
    with patch('open_data_scientist.codeagent.Client') as mock_client:
        # Configure the mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_response_obj
        
        # Create agent and run a task to generate a session
        agent = ReActDataScienceAgent(executor="internal", max_iterations=1)
        
        # Run the task - this should execute the mocked code and create a session
        _ = agent.run("test task")
        
        # Store the session ID
        session_id = agent.session_id
        assert session_id is not None
        
        # Delete the agent object to trigger cleanup
        del agent
        
        # Verify session was cleaned up by trying to delete it manually
        # This should fail because the session was already deleted by the destructor
        delete_result = delete_session_internal(session_id)
        assert delete_result["success"] is False
        assert "error" in delete_result
        
        # Test calling deletion twice - second call should also fail
        delete_result2 = delete_session_internal(session_id)
        assert delete_result2["success"] is False
        assert "error" in delete_result2 