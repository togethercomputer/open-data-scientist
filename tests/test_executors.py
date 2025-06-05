import pytest
from open_data_scientist.utils.executors import (
    execute_code_factory,
    execute_code_internal,
    execute_code_tci,
)


def test_factory_returns_internal_executor():
    """Test that factory returns the correct internal executor function."""
    executor = execute_code_factory("internal")
    assert executor == execute_code_internal


def test_factory_returns_tci_executor():
    """Test that factory returns the correct TCI executor function."""
    executor = execute_code_factory("tci")
    assert executor == execute_code_tci


def test_factory_raises_error_for_unknown_type():
    """Test that factory raises ValueError for unknown executor type."""
    with pytest.raises(ValueError, match="Unsupported code type: unknown"):
        execute_code_factory("unknown")


def test_internal_executor_simple_math():
    """Test internal executor with simple math expression."""
    result = execute_code_internal("2 + 2")

    assert result["status"] == "success"
    assert len(result["outputs"]) == 1
    assert result["outputs"][0]["type"] == "stdout"
    assert result["outputs"][0]["data"] == "4"
    assert result["session_id"] is not None
    assert result["errors"] == []


def test_internal_executor_print_statement():
    """Test internal executor with print statement."""
    result = execute_code_internal("print('hello world')")

    assert result["status"] == "success"
    assert len(result["outputs"]) == 1
    assert result["outputs"][0]["type"] == "stdout"
    assert result["outputs"][0]["data"] == "hello world"


def test_internal_executor_variable_assignment():
    """Test internal executor with variable assignment."""
    # First set the variable
    result1 = execute_code_internal("x = 10")
    session_id = result1["session_id"]

    # Then get the variable value
    result2 = execute_code_internal("x", session_id=session_id)

    assert result2["status"] == "success"
    assert len(result2["outputs"]) == 1
    assert result2["outputs"][0]["data"] == "10"


def test_internal_executor_with_session():
    """Test internal executor maintains session state."""
    # First call - set a variable
    result1 = execute_code_internal("x = 42")
    session_id = result1["session_id"]

    # Second call - use the variable from same session
    result2 = execute_code_internal("x * 2", session_id=session_id)

    assert result2["status"] == "success"
    assert result2["outputs"][0]["data"] == "84"
    assert result2["session_id"] == session_id


def test_internal_executor_error_handling():
    """Test internal executor handles errors properly."""
    result = execute_code_internal("undefined_variable")

    assert result["status"] == "failure"
    assert len(result["errors"]) > 0
    assert "not defined" in result["errors"][0]


def test_internal_executor_syntax_error():
    """Test internal executor handles syntax errors."""
    result = execute_code_internal("2 + + +")

    assert result["status"] == "failure"
    assert len(result["errors"]) > 0


def test_internal_executor_list_operations():
    """Test internal executor with list operations."""
    result = execute_code_internal("sum([1, 2, 3, 4, 5])")

    assert result["status"] == "success"
    assert result["outputs"][0]["data"] == "15"


def test_internal_executor_string_operations():
    """Test internal executor with string operations."""
    result = execute_code_internal("'hello'.upper()")

    assert result["status"] == "success"
    assert result["outputs"][0]["data"] == "HELLO"


def test_tci_executor_comprehensive():
    """Comprehensive TCI executor test using single session to reduce API costs."""
    # Test 1: Simple print statement
    result1 = execute_code_tci("print('Hello from TCI')")
    session_id = result1["session_id"]

    assert result1["status"] in ["success", "completed"]
    assert result1["session_id"] is not None
    assert "outputs" in result1
    assert len(result1["outputs"]) > 0

    # Check that the output contains our expected text
    output_found = False
    for output in result1["outputs"]:
        if "Hello from TCI" in str(output.get("data", "")):
            output_found = True
            break
    assert output_found, f"Expected 'Hello from TCI' in outputs: {result1['outputs']}"

    # Test 2: Math calculation (reuse session)
    result2 = execute_code_tci("result = 15 * 3; print(result)", session_id=session_id)

    assert result2["status"] in ["success", "completed"]
    assert result2["session_id"] == session_id

    # Check that the output contains the expected calculation result
    output_found = False
    for output in result2["outputs"]:
        if "45" in str(output.get("data", "")):
            output_found = True
            break
    assert output_found, f"Expected '45' in outputs: {result2['outputs']}"

    # Test 3: Session state persistence - set a variable
    result3 = execute_code_tci("my_var = 100", session_id=session_id)
    assert result3["status"] in ["success", "completed"]
    assert result3["session_id"] == session_id

    # Test 4: Use the variable from previous step
    result4 = execute_code_tci(
        "final_result = my_var + 50; print(final_result)", session_id=session_id
    )

    assert result4["session_id"] == session_id
    assert result4["status"] in ["success", "completed"]

    # Check that the output contains the expected calculation result
    output_found = False
    for output in result4["outputs"]:
        if "150" in str(output.get("data", "")):
            output_found = True
            break
    assert output_found, f"Expected '150' in outputs: {result4['outputs']}"

    # Test 5: Simple expression
    result5 = execute_code_tci("2 + 2", session_id=session_id)

    assert result5["status"] in ["success", "completed"]
    assert result5["session_id"] == session_id

    # Check that the output contains the expected result
    output_found = False
    for output in result5["outputs"]:
        if "4" in str(output.get("data", "")):
            output_found = True
            break
    assert output_found, f"Expected '4' in outputs: {result5['outputs']}"

    # Test 6: String operations
    result6 = execute_code_tci(
        "text = 'hello world'; print(text.upper())", session_id=session_id
    )

    assert result6["status"] in ["success", "completed"]
    assert result6["session_id"] == session_id

    # Check that the output contains the expected uppercase text
    output_found = False
    for output in result6["outputs"]:
        if "HELLO WORLD" in str(output.get("data", "")):
            output_found = True
            break
    assert output_found, f"Expected 'HELLO WORLD' in outputs: {result6['outputs']}"

    # Test 7: List operations
    result7 = execute_code_tci(
        "numbers = [1, 2, 3, 4, 5]; print(sum(numbers))", session_id=session_id
    )

    assert result7["status"] in ["success", "completed"]
    assert result7["session_id"] == session_id

    # Check that the output contains the expected sum
    output_found = False
    for output in result7["outputs"]:
        if "15" in str(output.get("data", "")):
            output_found = True
            break
    assert output_found, f"Expected '15' in outputs: {result7['outputs']}"


def test_factory_created_executors_are_callable():
    """Test that executors created by factory are callable."""
    internal_executor = execute_code_factory("internal")
    tci_executor = execute_code_factory("tci")

    assert callable(internal_executor)
    assert callable(tci_executor)

    # Test that we can actually call the internal executor
    result = internal_executor("1 + 1")
    assert result["status"] == "success"
