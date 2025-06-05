import argparse
from open_data_scientist.codeagent import ReActDataScienceAgent


# Competition name to data directory mapping
COMPETITIONS = {
    "spooky": "spooky",
    "jigsaw": "jigsaw",
}


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Run Kaggle competition task")
    parser.add_argument(
        "--competition",
        "-c",
        default="spooky",
        help="Competition name (default: spooky)",
    )

    args = parser.parse_args()

    # Set data directory based on competition
    competition_key = args.competition.lower()
    data_dir = f"/app/{COMPETITIONS[competition_key]}"

    PROMPT = f"""You are a data scientist tasked with solving a Kaggle competition.

TASK: You have been given a folder containing a Kaggle competition dataset. Your job is to:

You will find the dataset in the {data_dir} folder.
1) Read and understand the task by examining any readme or description files in the folder
2) Analyze the provided data files (train.csv, test.csv, sample_submission.csv, etc.)
3) Build a solution to solve the competition task
4) Generate a submission file in the exact format required by the competition


INSTRUCTIONS:
1) Start by examining the data structure and understanding the task
2) Develop your approach based on the problem type (classification, regression, NLP, etc.)
3) Generate predictions for all entries in test.csv
4) Create a submission file named "{data_dir}/submission.csv" in the correct format as shown in sample_submission.csv

Please solve this step by step and create the final submission file.
"""

    print(f"Starting {args.competition} competition task...")
    print(f"Data directory: {data_dir}")

    agent = ReActDataScienceAgent(executor="internal")
    result = agent.run(PROMPT)
    print(result)

    print(
        f"Task completed. Check the {data_dir} folder for the submission.csv file created by the agent."
    )


if __name__ == "__main__":
    main()
