from open_data_scientist.utils.strings import get_execution_summary


def test_truncation_with_long_output():
    """Test that long outputs are properly truncated."""
    long_output = "This is a very long output. " * 1000  # ~25,000 characters
    
    execution_result = {
        "status": "success",
        "result": long_output,
        "outputs": [{"type": "stdout", "data": long_output}]
    }
    
    result = get_execution_summary(execution_result, max_chars_before_truncation=1000)
    
    # Should be truncated
    assert len(result) < len(long_output)
    assert "characters truncated" in result
    assert "Execution status: success" in result


def test_no_truncation_with_short_output():
    """Test that short outputs are not truncated."""
    short_output = "This is a short output."
    
    execution_result = {
        "status": "success",
        "result": short_output,
        "outputs": [{"type": "stdout", "data": short_output}]
    }
    
    result = get_execution_summary(execution_result, max_chars_before_truncation=1000)
    
    # Should not be truncated
    assert "characters truncated" not in result
    assert short_output in result
    assert "Execution status: success" in result


def test_truncation_with_errors():
    """Test that error outputs are truncated correctly."""
    long_error = "This is a very long error message. " * 200  # ~5,000 chars
    
    execution_result = {
        "status": "failed",
        "errors": [long_error]
    }
    
    result = get_execution_summary(execution_result, max_chars_before_truncation=1000)
    
    # Should be truncated
    assert len(result) < len(long_error)
    assert "characters truncated" in result
    assert "Execution status: failed" in result 


def test_truncation_respects_the_configured_limit():
    """Head and tail together keep max_chars_before_truncation characters, and the count is exact."""
    long_output = "x" * 2500

    execution_result = {
        "status": "success",
        "outputs": [{"type": "stdout", "data": long_output}],
    }

    full_length = len(f"Execution status: success\nText output:\n{long_output}")
    result = get_execution_summary(execution_result, max_chars_before_truncation=1000)

    assert f"({full_length - 1000} characters truncated" in result
    head, _, rest = result.partition("... (")
    tail = rest.rpartition(") ...")[2]
    assert len(head) + len(tail) == 1000

    result = get_execution_summary(execution_result, max_chars_before_truncation=10_000)
    assert "characters truncated" not in result
