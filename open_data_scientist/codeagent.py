import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from together import Client

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.rule import Rule
from open_data_scientist.utils.strings import PROMPT_TEMPLATE, get_execution_summary
from open_data_scientist.utils.executors import (
    execute_code_factory,
    upload_file_internal,
    create_tci_session_with_data,
    delete_session_internal,
)
from open_data_scientist.utils.strings import print_rich_execution_result

console = Console()

# TODO: make these into a config file
reasoning_model = "deepseek-ai/DeepSeek-V3"
max_iterations = 20
temperature = 0.1
max_chars_before_truncation = 40000
max_tokens = 20000


class SessionSwapError(Exception):
    pass



class ReActDataScienceAgent:
    def __init__(
        self,
        session_id: Optional[str] = None,
        model: str = reasoning_model,
        max_iterations: int = max_iterations,
        executor: str = "internal",
        data_dir: Optional[str] = None,
        trace_path: Optional[str] = None,
    ):
        self.client = Client()
        self.session_id = session_id
        self.model = model
        self.max_iterations = max_iterations
        self.data_dir = data_dir
        self.executor_type = executor
        self.trace_path = trace_path
        self.log_path = self._derive_log_path(trace_path)

        self.system_prompt = PROMPT_TEMPLATE

        # we will start adding the system prompt here
        self.history = [{"role": "system", "content": self.system_prompt["DATA_SCIENCE_AGENT"]}]

        if self.data_dir:
            if executor == "tci":
                self.session_id = create_tci_session_with_data(data_dir=self.data_dir)
            else:
                upload_file_internal(self.data_dir)

        self.executor: Callable = execute_code_factory(executor)

    def __del__(self):
        """Clean up session when object is deleted"""
        if hasattr(self, 'session_id') and self.session_id:
            try:
                delete_session_internal(self.session_id)
            except Exception:
                pass

    def _derive_log_path(self, trace_path: Optional[str]) -> Optional[str]:
        if not trace_path:
            return None
        return str(Path(trace_path).with_suffix(".md"))

    def _record_event(self, event: dict) -> None:
        if not self.trace_path:
            return
        event.setdefault(
            "timestamp",
            datetime.now().isoformat(timespec="seconds"),
        )
        if self.trace_path:
            with open(self.trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        if self.log_path is not None:
            self._run_events.append(event)

    def _extract_image_filenames(self, observation: str) -> list[str]:
        """
        Extract saved image filenames from observation text.
        Observation strings include 'saved as: filename.png' when plots are saved.
        """
        return re.findall(r"saved as: ([\w._-]+\.png)", observation)

    def _render_log(self) -> None:
        if not self.log_path or not self._run_events:
            return
        log_file = Path(self.log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not log_file.exists() or log_file.stat().st_size == 0

        with open(log_file, "a", encoding="utf-8") as handle:
            if is_new_file:
                handle.write("# ReAct Run Log\n\n")

            for event in self._run_events:
                event_type = event.get("type")
                if event_type == "query":
                    handle.write("\n## Query\n")
                    handle.write(f"- Task: {event.get('query', '')}\n")
                    handle.write(f"- Model: {event.get('model', '')}\n")
                    handle.write(f"- Executor: {event.get('executor', '')}\n")
                    handle.write(f"- Session: {event.get('session_id', 'None')}\n")
                    handle.write(f"- Data directory: {event.get('data_dir', 'None')}\n")
                    handle.write(f"- Started: {event.get('started_at', '')}\n\n")
                elif event_type == "step":
                    handle.write(f"### Step {event.get('iteration', '')}\n\n")
                    handle.write("**Thought**\n")
                    handle.write(f"{event.get('thought', '')}\n\n")
                    handle.write("**Code**\n")
                    handle.write(f"```python\n{event.get('code', '')}\n```\n\n")
                    handle.write("**Observation**\n")
                    handle.write(f"{event.get('observation', '')}\n\n")
                    images = event.get("images", [])
                    if images:
                        handle.write("Images:\n")
                        for image in images:
                            handle.write(f"![{image}]({image})\n")
                        handle.write("\n")
                elif event_type == "error":
                    handle.write(f"### Error at step {event.get('iteration', '')}\n\n")
                    handle.write(f"{event.get('error', '')}\n\n")
                elif event_type == "final_answer":
                    handle.write("\n**Final Answer**\n\n")
                    handle.write(f"{event.get('answer', '')}\n\n")
                    images = event.get("images", [])
                    if images:
                        handle.write("Images:\n")
                        for image in images:
                            handle.write(f"![{image}]({image})\n")
                        handle.write("\n")
                elif event_type == "max_iterations":
                    handle.write(
                        f"WARNING: Maximum iterations reached ({event.get('max_iterations', '')})\n\n"
                    )
                    handle.write(f"{event.get('result', '')}\n\n")
                elif event_type == "run_end":
                    handle.write(f"Finished: {event.get('finished_at', '')}\n\n")
                    handle.write("---\n\n")

    def final_anwer_execution(self, final_answer: str, session_id: str | None):
        """Execute all Python code blocks in the final answer and replace them with their results"""
        code_blocks = re.finditer(r"```python\n(.*?)\n```", final_answer, re.DOTALL)
        
        for match in code_blocks:
            code = match.group(1)
            result = self.executor(code, session_id)
            
            execution_summary = get_execution_summary(result, max_chars_before_truncation)
            
            final_answer = final_answer.replace(match.group(0), execution_summary)
            
        return final_answer


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def llm_call(self) -> str:
        """Make a call to the language model with retry logic"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
            stream=False,
        )
        return response.choices[0].message.content  # type: ignore

    def parse_response(self):
        """Parse the LLM response and extract thought and action input"""
        response = self.llm_call()

        # Try to find code block - either direct content or markdown inside <code>
        code_match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
            # Check if the content inside <code> is a markdown block
            markdown_match = re.search(r"```(?:python)?\s*(.*?)\s*```", code_content, re.DOTALL)
            action_input = markdown_match.group(1).strip() if markdown_match else code_content
            
            # Try to find thought
            thought_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else "The assistant didn't provide a proper thought section."
            
            return thought, action_input

        # Try to find final answer
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            return final_answer, None

        console.print(
            f"[bold red]ERROR:[/bold red] No valid tags found in response:\n{response}"
        )
        thought = "I need to be careful with the format in the response"
        action_input = "print('Error: Format not followed by the assistant, please use <think>, <code>, or <answer> tags.')"
        return thought, action_input

    def run(self, user_input: str):
        """Execute the main ReAct reasoning and acting loop"""
        self.history.append({"role": "user", "content": user_input})

        current_iteration = 0
        result = None
        started_at = datetime.now().isoformat(timespec="seconds")
        self._run_events = []

        self._record_event(
            {
                "type": "query",
                "query": user_input,
                "model": self.model,
                "executor": self.executor_type,
                "session_id": self.session_id,
                "data_dir": self.data_dir,
                "started_at": started_at,
            }
        )

        # Rich startup display
        startup_text = f"🚀 Starting ReAct Data Science Agent\n📝 Task: {user_input}"
        startup_panel = Panel(
            startup_text,
            title="🤖 ReAct Data Science Agent",
            border_style="bold blue",
            expand=False,
            width=80,
        )
        console.print(startup_panel)
        console.print()

        while current_iteration < self.max_iterations:
            try:
                result, action_input = self.parse_response()

                if action_input is None:
                    result = self.final_anwer_execution(result, self.session_id)
                    
                    # Add final answer to history
                    self.history.append({"role": "assistant", "content": f"<answer>\n{result}\n</answer>"})

                    final_images = self._extract_image_filenames(result)
                    self._record_event(
                        {
                            "type": "final_answer",
                            "answer": result,
                            "images": final_images,
                            "session_id": self.session_id,
                        }
                    )
                    
                    final_panel = Panel(
                        result,
                        title="🎯 Final Answer",
                        border_style="bold green",
                        expand=False,
                        width=80,
                    )
                    console.print(final_panel)
                    # TODO this is ugly, we should have a better way to handle this
                    break

                thought = result

                # Thought panel
                thought_panel = Panel(
                    thought,
                    title=f"🤔 Thought (Iteration {current_iteration + 1})",
                    border_style="blue",
                    expand=False,
                    width=80,
                )
                console.print(thought_panel)

                # Action panel with syntax highlighting
                action_syntax = Syntax(
                    action_input, "python", theme="monokai", line_numbers=True
                )
                action_panel = Panel(
                    action_syntax,
                    title="🛠️ Action",
                    border_style="yellow",
                    expand=False,
                    width=80,
                )
                console.print(action_panel)

                # Execute the code
                execution_result = self.executor(action_input, self.session_id)

                if execution_result and "session_id" in execution_result:
                    new_session_id = execution_result["session_id"]
                    if self.session_id is not None and new_session_id != self.session_id:
                        raise SessionSwapError("Session ID changed unexpectedly")
                    self.session_id = new_session_id

                # Display results
                print_rich_execution_result(
                    execution_result, f"Result {self.session_id}", "📊"
                )

                # Get summary for agent's history
                execution_summary = get_execution_summary(execution_result, max_chars_before_truncation)

                # Add to conversation history. We use the "user" role for the observation content.
                # You could alos use other tools.
                add_to_history = (
                    f"<think>\n{thought}\n</think>\n<code>\n```python\n{action_input}\n```\n</code>"
                )
                self.history.append({"role": "assistant", "content": add_to_history})
                self.history.append(
                    {"role": "user", "content": f"Observation: {execution_summary}"}
                )

                images = self._extract_image_filenames(execution_summary)
                self._record_event(
                    {
                        "type": "step",
                        "iteration": current_iteration + 1,
                        "thought": thought,
                        "code": action_input,
                        "observation": execution_summary,
                        "images": images,
                        "session_id": self.session_id,
                    }
                )

                current_iteration += 1
                console.print(Rule(style="dim"))

            except SessionSwapError as e:
                self._record_event(
                    {
                        "type": "error",
                        "iteration": current_iteration + 1,
                        "error": str(e),
                        "session_id": self.session_id,
                    }
                )
                self._render_log()
                console.print(
                    f"💀 [bold red]FATAL ERROR: Session ID changed unexpectedly! This is often due to long running tasks.[/bold red] {str(e)}"
                )
                console.print("[bold red]Killing the program to prevent data corruption.[/bold red]")
                sys.exit(1)
            except Exception as e:
                self._record_event(
                    {
                        "type": "error",
                        "iteration": current_iteration + 1,
                        "error": str(e),
                        "session_id": self.session_id,
                    }
                )
                console.print(
                    f"❌ [bold red]Error in iteration {current_iteration + 1}:[/bold red] {str(e)}"
                )
                # Add error to history and continue
                self.history.append(
                    {
                        "role": "user",
                        "content": f"Error occurred: {str(e)}. Please try a different approach.",
                    }
                )
                current_iteration += 1
        
        if current_iteration >= self.max_iterations:
            console.print(
                f"⚠️ [bold yellow]Maximum iterations ({self.max_iterations}) reached without completion[/bold yellow]"
            )
            result = result or "Task incomplete - maximum iterations reached"
            self._record_event(
                {
                    "type": "max_iterations",
                    "max_iterations": self.max_iterations,
                    "result": result,
                    "session_id": self.session_id,
                }
            )
        finished_at = datetime.now().isoformat(timespec="seconds")
        self._record_event(
            {
                "type": "run_end",
                "finished_at": finished_at,
                "session_id": self.session_id,
            }
        )
        self._render_log()

        return result


def main():
    """
    Simple main function demonstrating the ReAct Data Science Agent with internal upload.

    This will:
    1. Upload all files from a specified data directory
    2. Run a basic data exploration experiment
    """
    import os

    # Configure the data directory - change this to your data folder
    data_directory = "/Users/federico/projects/open-data-scientist/eval/kaggle_data/jigsaw"  # Change this path to your data folder

    # Check if data directory exists
    if not os.path.exists(data_directory):
        console.print(
            f"[bold red]Error:[/bold red] Data directory '{data_directory}' not found!"
        )
        console.print(
            "[yellow]Please create the directory and add some data files (CSV, JSON, TXT, etc.)[/yellow]"
        )
        console.print(
            "[yellow]Or change the data_directory variable to point to your data folder[/yellow]"
        )
        return

    # Create the agent with internal executor and data directory
    console.print(
        f"[bold green]🔄 Initializing agent with data from:[/bold green] {data_directory}"
    )

    agent = ReActDataScienceAgent(
        executor="tci",  # Use our internal interpreter service
        data_dir=data_directory,  # This will automatically upload all files
        max_iterations=10,
    )

    # Run a simple data exploration task
    task = """
    Please explore the uploaded data files. Start by:
    1. List all available files in the current directory
    2. For any CSV files found, load them and show basic information (shape, columns, first few rows)
    3. Provide a summary of what data is available
    """

    console.print("[bold blue]🎯 Running task:[/bold blue] Data exploration")

    result = agent.run(task)

    console.print("\n[bold green]✅ Task completed![/bold green]")
    return result


if __name__ == "__main__":
    main()
