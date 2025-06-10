import re
import sys
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.rule import Rule
from open_data_scientist.utils.strings import PROMPT_TEMPLATE, get_execution_summary
from open_data_scientist.utils.executors import (
    execute_code_factory,
    upload_file_internal,
    create_tci_session_with_data,
)
from open_data_scientist.utils.strings import print_rich_execution_result

from together import Client

console = Console()

reasoning_model = "deepseek-ai/DeepSeek-V3"
max_iterations = 20
temperature = 0.1


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
    ):
        self.client = Client()
        self.session_id = session_id
        self.model = model
        self.max_iterations = max_iterations
        self.data_dir = data_dir

        self.system_prompt = PROMPT_TEMPLATE

        # we will start adding the system prompt here
        self.history = [{"role": "system", "content": self.system_prompt["DATA_SCIENCE_AGENT"]}]

        if self.data_dir:
            if executor == "tci":
                self.session_id = create_tci_session_with_data(data_dir=self.data_dir)
            else:
                upload_file_internal(self.data_dir)

        self.executor: Callable = execute_code_factory(executor)

    def llm_call(self) -> str:
        """Make a call to the language model"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=temperature,
            max_tokens=20000,
            stream=False,
        )
        return response.choices[0].message.content  # type: ignore

    def parse_response(self):
        """Parse the LLM response and extract thought and action input"""
        response = self.llm_call()

        if "Action Input:" in response:
            if "Thought:" in response:
                thought = (
                    response.split("Thought:")[1].split("Action Input:")[0].strip()
                )
            else:
                thought = "The assistant didn't provide a proper thought section."

            action_input_section = response.split("Action Input:")[1]
            code_match = re.search(
                r"```(?:python)?\s*(.*?)\s*```", action_input_section, re.DOTALL
            )

            if code_match:
                action_input = code_match.group(1).strip()
                return thought, action_input
            else:
                console.print(
                    f"[bold red]ERROR:[/bold red] No code block found after Action Input:\n{response}"
                )
                action_input = "print('Error: No code block found after Action Input')"
                return thought, action_input

        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[1].strip()
            return final_answer, None

        console.print(
            f"[bold red]ERROR:[/bold red] No Action Input or Final Answer found in response:\n{response}"
        )
        thought = "The assistant didn't follow the ReAct format properly."
        action_input = "print('Error: Format not followed by the assistant, please follow the format and try again. You need to use one of the two options: Action Input or Final Answer.')"
        return thought, action_input
    


    def run(self, user_input: str):
        """Execute the main ReAct reasoning and acting loop"""
        self.history.append({"role": "user", "content": user_input})

        current_iteration = 0

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
                    # Final answer panel
                    final_panel = Panel(
                        result,
                        title="🎯 Final Answer",
                        border_style="bold green",
                        expand=False,
                        width=80,
                    )
                    console.print(final_panel)
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
                execution_summary = get_execution_summary(execution_result)

                # Add to conversation history. We use the "user" role for the observation content.
                # You could alos use other tools.
                add_to_history = (
                    f"Thought: {thought}\nAction Input:```python\n{action_input}\n```"
                )
                self.history.append({"role": "assistant", "content": add_to_history})
                self.history.append(
                    {"role": "user", "content": f"Observation: {execution_summary}"}
                )

                current_iteration += 1
                console.print(Rule(style="dim"))

            except SessionSwapError as e:
                console.print(
                    f"💀 [bold red]FATAL ERROR: Session ID changed unexpectedly! This is often due to long running tasks.[/bold red] {str(e)}"
                )
                console.print("[bold red]Killing the program to prevent data corruption.[/bold red]")
                sys.exit(1)
            except Exception as e:
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
