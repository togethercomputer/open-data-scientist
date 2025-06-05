import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from open_data_scientist.codeagent import ReActDataScienceAgent


@dataclass(frozen=True)
class TaskResult:
    tid: str
    question: str
    answer: str | None = None
    llm_answer: str | None = None
    is_correct: bool | None = None
    reasoning_trace: str | None = None


def process_task(task, submit, data_dir: str | None = None):
    tid = task["task_id"]
    question = task["question"]
    guidelines = task["guidelines"]
    answer = None
    is_correct = None

    PROMPT = f"""You are an expert data analyst tasked with answering factoid questions by analyzing the following dataset files:

    AVAILABLE FILES:
    /app/downloaded_data/data/context/acquirer_countries.csv
    /app/downloaded_data/data/context/fees.json
    /app/downloaded_data/data/context/manual.md
    /app/downloaded_data/data/context/merchant_category_codes.csv
    /app/downloaded_data/data/context/merchant_data.json
    /app/downloaded_data/data/context/payments-readme.md
    /app/downloaded_data/data/context/payments.csv

    IMPORTANT: Always use the full/absolute paths shown above to access files. Relative paths will not work.

    ANALYSIS PROCESS:
    1) CRITICAL FIRST STEP: You MUST thoroughly read and internalize the manual.md file COMPLETELY before proceeding.
       - The manual contains domain-specific definitions that are ESSENTIAL for correct interpretation
       - Terms like "fees", "transactions", and other concepts have specialized meanings in this context
       - Misunderstanding these definitions will GUARANTEE incorrect answers
       - Create a mental model of how all concepts interrelate based on the manual's definitions
       - Pay special attention to any hierarchical relationships, exclusions, or conditional statements

    2) When reading the question, map it back to the exact terminology and definitions from the manual
       - Do NOT rely on your general knowledge about these terms
       - The correct answer depends on using the EXACT definitions from the manual
       - Identify which specific section of the manual is most relevant to the question

    3) FOR COMPLEX MULTI-STEP QUESTIONS: Break down the question into logical sub-components
       - Identify all the specific filters needed (merchant names, time periods, fee IDs, etc.)
       - Determine the sequence of operations required (filter → calculate → aggregate → compare)
       - For hypothetical scenarios (e.g., "what if fee changed to X"), clearly identify:
         * Current state calculation
         * Hypothetical state calculation  
         * Delta/difference calculation
       - For time-based questions, ensure you understand the exact date ranges and formatting
       - For merchant-specific questions, verify exact merchant name matching (case-sensitive)
       - For fee-related questions, distinguish between fee applicability vs. fee amounts vs. fee calculations

    4) Next, read the payments-readme.md file to understand the payment data structure and relevant terminology.

    5) For each additional file you need to access:
       - For CSV files: Check the column headers first to understand the data structure
       - For JSON files: Examine the schema by looking at a small sample (first few entries)
       - For text/markdown files: Read through the entire content for critical information

    6) When working with large files, start by understanding their structure before attempting to process all the data.

    7) Data validation and quality checks:
       - Check for missing values, duplicates, or data inconsistencies
       - Verify data types match expectations (strings, numbers, dates, etc.)
       - Look for outliers or anomalies that might affect your analysis
       - Cross-reference data between files to ensure consistency

    8) VERIFICATION STEP: Before finalizing your answer, always:
       - Re-read the relevant sections of the manual to confirm your interpretation
       - Double-check your calculations and data aggregations
       - For multi-step calculations, verify each intermediate result makes sense
       - For time-based filtering, confirm you're using the correct date format and range
       - For merchant-specific queries, verify exact name matches
       - For fee calculations, confirm you're applying the right fee rules and formulas
       - Verify your answer makes logical sense given the context
       - Ensure you're answering the exact question asked (not a related but different question)

    QUESTION TO ANSWER:
    {question}

    ANSWER GUIDELINES:
    {guidelines}

    CRITICAL REQUIREMENTS:
    - Be precise with numerical answers (include appropriate decimal places, units, etc.)
    - If calculations are involved, show your work clearly step-by-step
    - For complex multi-step problems, show all intermediate calculations
    - If the answer requires aggregation, explicitly state what you're aggregating
    - For categorical answers, use exact terminology from the manual/data
    - If data is missing or incomplete, state this clearly rather than guessing
    - For hypothetical scenarios, clearly distinguish current vs. hypothetical calculations
    - STRICTLY ADHERE TO THE GUIDELINES for formatting your output
    
    FINAL ANSWER FORMAT:
    After your analysis, provide your final answer in the exact format specified in the ANSWER GUIDELINES. You might want to generate the formatted answer in python first and then copy the formatted answer to your Final Answer section.
    
    If you encounter any errors accessing files or processing data, clearly state what went wrong rather than providing a guess.
    """

    print(f"Processing question: {question[:50]}...")

    agent = ReActDataScienceAgent(executor="internal")

    try:
        llm_answer = agent.run(PROMPT)
    except Exception as e:
        print(f"Task {tid} generated an exception: {e}")
        llm_answer = "Error: Task failed with exception"

    reasoning_trace = "Not available"

    if not submit:
        answer = task["answer"]
        is_correct = (
            answer == llm_answer
            if llm_answer != "Error: Task failed with exception"
            else False
        )

    return TaskResult(
        tid=tid,
        question=question,
        answer=answer,
        llm_answer=llm_answer,
        is_correct=is_correct,
        reasoning_trace=reasoning_trace,
    )


def write_jsonl(data: list[dict], filepath: Path) -> None:
    """Write a list of dictionaries to a JSONL file."""
    # Ensure the directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def main(
    test_first_only=False,
    submit=False,
    data_dir=None,
    which_split="dev",
    skip_hard=False,
):
    # Load the dataset
    ds = load_dataset("adyen/DABstep", "tasks")

    dataset = ds[which_split]

    # Store hard tasks before filtering if we're skipping and submitting
    skipped_tasks = []
    if skip_hard and submit:
        skipped_tasks = [task for task in dataset if task.get("level") == "hard"]

    if skip_hard:
        dataset = dataset.filter(lambda example: example.get("level") != "hard")

    if test_first_only:
        dataset = dataset.select([0, 1, 2])

    number_of_examples = len(dataset)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {
            executor.submit(process_task, task, submit, data_dir): task
            for task in dataset
        }
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed task: {result.is_correct}")
            except Exception as e:
                # This should rarely happen now since exceptions are handled in process_task
                task = future_to_task[future]
                print(f"Unexpected error in task execution: {e}")

    if submit:
        results_to_submit = [
            {
                "task_id": result.tid,
                "agent_answer": str(result.llm_answer),
                "reasoning_trace": str(result.reasoning_trace),
            }
            for result in results
        ]

        # Add skipped hard tasks
        for task in skipped_tasks:
            results_to_submit.append(
                {
                    "task_id": task["task_id"],
                    "agent_answer": "Error",
                    "reasoning_trace": "skipped",
                }
            )

        write_jsonl(
            results_to_submit,
            Path(__file__).parent.parent / "submissions" / "DABstep" / "results.jsonl",
        )

    correct_count = sum(r.is_correct for r in results if r.is_correct is not None)
    print(
        f"\nResults: {correct_count}/{number_of_examples} correct answers ({correct_count / number_of_examples * 100:.2f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DABstep evaluation")
    parser.add_argument(
        "--test-first-only", action="store_true", help="Test only the first example"
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit the results to the leaderboard"
    )
    parser.add_argument(
        "--which-split",
        type=str,
        default="dev",
        help="Which split to use (default: dev)",
    )
    parser.add_argument(
        "--skip-hard", action="store_true", help="Skip examples with level=hard"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing data files to upload to the agent session",
    )
    args = parser.parse_args()

    main(
        test_first_only=args.test_first_only,
        submit=args.submit,
        data_dir=args.data_dir,
        which_split=args.which_split,
        skip_hard=args.skip_hard,
    )
