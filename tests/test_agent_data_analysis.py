from open_data_scientist.codeagent import ReActDataScienceAgent


def test_agent_analyze_auto_mpg_csv_end_to_end():
    """End-to-end test: Agent loads auto-mpg CSV and analyzes its columns."""

    # Create the agent with internal executor
    agent = ReActDataScienceAgent(executor="internal", max_iterations=3)

    # Give the agent a task to analyze the CSV data
    user_request = """Load the auto-mpg dataset from this URL: https://huggingface.co/datasets/scikit-learn/auto-mpg/raw/main/auto-mpg.csv
    
    Then tell me:
    1. What columns does the dataset have?
    2. What is the shape of the dataset?
    3. What are the data types of each column?"""

    # Run the agent
    result = agent.run(user_request)

    # Verify the agent completed successfully and found the mpg column
    assert isinstance(result, str)
    assert len(result) > 0
    assert "cylinders" in result.lower()
