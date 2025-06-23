import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path
import random
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
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

ANALYTICAL GUIDELINES:
- When asked to find values across multiple applicable rules or data points, ensure you include ALL relevant items in your analysis
- Do not arbitrarily select a single item when multiple items apply
- Count all matching items, not just the first one found
- Pay special attention to null values in rule definitions - they mean "applies to all values" not "no match"
- When filtering rules, be less restrictive rather than more restrictive
- Consider that some entities may not have specific rules and may use default/fallback rules
- Verify your rule matching logic by checking if you're finding reasonable numbers of applicable rules
- When you find 0 applicable rules, reconsider your filtering criteria - this often indicates overly restrictive logic
- When processing multiple data points, verify that you're including all relevant items
- When comparing options across different characteristics, ensure you're using the correct rules for each option
- Don't assume that the lowest calculated value is automatically the correct answer - verify the rules actually apply
- Consider all relevant characteristics when determining rule applicability
- Cross-reference rules with actual data to ensure realistic scenarios

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

    print(f"Processing question: {question[:100]}...")

    agent = ReActDataScienceAgent(executor="internal", max_iterations=30)

    try:
        llm_answer = agent.run(PROMPT)
        
        # Extract reasoning traces from agent history
        reasoning_trace = json.dumps(agent.history)
            
    except Exception as e:
        print(f"Task {tid} generated an exception: {e}")
        llm_answer = "Error: Task failed with exception"
        reasoning_trace = f"Error occurred: {str(e)}"

    # Always compute correctness if answer is available, regardless of submit flag
    if "answer" in task:
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
    submit=False,
    data_dir=None,
    which_split="dev",
    skip_hard=False,
    reduced_test=False,
):
    if skip_hard and reduced_test:
        raise ValueError("Cannot use both --skip-hard and --reduced-test at the same time")

    # Load the dataset
    ds = load_dataset("adyen/DABstep", "tasks")

    dataset = ds[which_split]

    # Store hard tasks before filtering if we're skipping and submitting
    skipped_tasks = []
    if skip_hard:
        skipped_tasks = [task for task in dataset if task.get("level") == "hard"]
        dataset = dataset.filter(lambda example: example.get("level") != "hard")
    elif reduced_test:
        dataset = dataset.shuffle(seed=42)
        easy_tasks = dataset.filter(lambda x: x["level"] == "easy")
        hard_tasks = dataset.filter(lambda x: x["level"] == "hard")

        # Sample 20 tasks from each difficulty level
        sampled_easy = easy_tasks.select(range(20))  
        sampled_hard = hard_tasks.select(range(20))  

        sampled_ids = set()
        for task in sampled_easy:
            sampled_ids.add(task["task_id"])
        for task in sampled_hard:
            sampled_ids.add(task["task_id"])

        skipped_tasks = [task for task in dataset if task["task_id"] not in sampled_ids]
        dataset = concatenate_datasets([sampled_easy, sampled_hard])
        dataset = dataset.shuffle(seed=42)
    else:
        print("Running all tasks")

    number_of_examples = len(dataset)
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(process_task, task, submit, data_dir): task
            for task in dataset
        }
        
        with tqdm(total=number_of_examples, desc="Processing tasks") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    status = "✓" if result.is_correct else "✗"
                    pbar.set_postfix_str(f"Task {result.tid}: {status}")
                    pbar.update(1)
                except Exception as e:
                    # This should rarely happen now since exceptions are handled in process_task
                    task = future_to_task[future]
                    pbar.set_postfix_str(f"Task {task['task_id']}: ERROR - {str(e)[:30]}...")
                    pbar.update(1)

    if submit:
        results_to_submit = [
            {
                "task_id": result.tid,
                "agent_answer": str(result.llm_answer),
                "reasoning_trace": str(result.reasoning_trace),
                "correct_answer": str(result.answer) if result.answer is not None else None,
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
                    "correct_answer": task.get("answer"),
                }
            )

        write_jsonl(
            results_to_submit,
            Path(__file__).parent.parent / "submissions" / "DABstep" / "results.jsonl",
        )

    # Calculate results for tasks that have answers available
    tasks_with_answers = [r for r in results if r.is_correct is not None]
    correct_count = sum(r.is_correct for r in tasks_with_answers)
    total_with_answers = len(tasks_with_answers)
    
    if total_with_answers > 0:
        print(
            f"\nResults: {correct_count}/{total_with_answers} correct answers ({correct_count / total_with_answers * 100:.2f}%)"
        )
        if total_with_answers < number_of_examples:
            print(f"Note: {number_of_examples - total_with_answers} tasks did not have answer keys available")
    else:
        print(f"\nProcessed {number_of_examples} tasks (no answer keys available for accuracy calculation)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DABstep evaluation")
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
        "--reduced-test", action="store_true", help="Sample 20 easy and 20 hard tasks"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing data files to upload to the agent session",
    )
    args = parser.parse_args()

    main(
        submit=args.submit,
        data_dir=args.data_dir,
        which_split=args.which_split,
        skip_hard=args.skip_hard,
        reduced_test=args.reduced_test,
    )
