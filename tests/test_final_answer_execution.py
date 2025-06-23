from open_data_scientist.codeagent import ReActDataScienceAgent


def test_final_answer_execution():
    """Test that Python code blocks in final answer are executed and replaced with results."""
    agent = ReActDataScienceAgent(executor="internal")
    
    final_answer = """Here is my analysis:

Let me calculate some values:

```python
import math
x = 10
y = 20
z = x + y
radius = 5
area = math.pi * radius ** 2
print(f"x = {x}, y = {y}, z = {z}")
print(f"Circle with radius {radius} has area: {area:.2f}")
```

The calculations are complete."""
    
    result = agent.final_anwer_execution(final_answer, "test-session")
    
    print("--------------------------------")
    print(result)
    print("--------------------------------")
    # Check that the code block was replaced with execution summary
    assert "```python" not in result
    assert "x = 10, y = 20, z = 30" in result
    assert "Circle with radius 5 has area: 78.54" in result
    assert "Execution status: success" in result 