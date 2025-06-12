#!/usr/bin/env python3
"""
Command Line Interface for the ReAct Data Science Agent
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from open_data_scientist.utils.writer import _write_report

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from open_data_scientist.codeagent import ReActDataScienceAgent

console = Console()


def get_data_directory(data_dir: Optional[str]) -> Optional[str]:
    """
    Handle data directory selection with user confirmation for current directory.
    """
    if data_dir is None:
        # No data directory specified, use current directory but ask for confirmation
        current_dir = os.getcwd()

        console.print("\n[yellow]No data directory specified.[/yellow]")
        console.print(f"[blue]Current directory:[/blue] {Path(current_dir).name}")

        # Show files in current directory
        files = list(Path(current_dir).iterdir())
        data_files = [
            f
            for f in files
            if f.is_file()
            and f.suffix.lower() in [".csv", ".json", ".txt", ".py", ".xlsx", ".xls"]
        ]

        if data_files:
            console.print(
                f"\n[green]Found {len(data_files)} potential data files:[/green]"
            )
            for file in data_files[:10]:  # Show max 10 files
                console.print(f"  • {file.name}")
            if len(data_files) > 10:
                console.print(f"  ... and {len(data_files) - 10} more files")
        else:
            console.print(
                "\n[yellow]No obvious data files found in current directory.[/yellow]"
            )

        use_current = Confirm.ask(
            "\n[bold]Important: Do you want to upload files from the current directory?[/bold]",
            default=False,
        )

        if use_current:
            return current_dir
        else:
            console.print("[yellow]Proceeding without uploading files.[/yellow]")
            return None
    else:
        # Data directory specified, validate it exists
        if not os.path.exists(data_dir):
            console.print(
                f"[bold red]Error:[/bold red] Data directory '{data_dir}' not found!"
            )
            sys.exit(1)

        if not os.path.isdir(data_dir):
            console.print(
                f"[bold red]Error:[/bold red] '{data_dir}' is not a directory!"
            )
            sys.exit(1)

        return data_dir


def validate_executor(executor: str) -> str:
    """Validate executor choice"""
    valid_executors = ["tci", "internal"]
    if executor not in valid_executors:
        console.print(
            f"[bold red]Error:[/bold red] Invalid executor '{executor}'. Must be one of: {', '.join(valid_executors)}"
        )
        sys.exit(1)
    return executor


def show_configuration(args) -> None:
    """Display the current configuration in a nice table"""
    table = Table(title="🤖 ReAct Data Science Agent Configuration")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Model", args.model)
    table.add_row("Max Iterations", str(args.iterations))
    table.add_row("Executor", args.executor)
    table.add_row(
        "Data Directory", args.data_dir or "Current directory (with confirmation)"
    )

    console.print(table)
    console.print()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🤖 ReAct Data Science Agent - AI-powered data analysis assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with TCI (cloud) executor
  open-data-scientist

  # Use specific model and more iterations
  open-data-scientist --model "deepseek-ai/DeepSeek-V3" --iterations 15

  # Use local Docker executor with specific data directory
  open-data-scientist --executor internal --data-dir /path/to/data

  # Interactive mode with custom settings
  open-data-scientist --model "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" --iterations 20 --data-dir ./my_data

Execution Modes:
  tci      - Cloud execution via Together AI (requires API key)
  internal - Local Docker execution (requires docker-compose setup)
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        default="deepseek-ai/DeepSeek-V3",
        help="Language model to use (default: deepseek-ai/DeepSeek-V3)",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=20,
        help="Maximum number of reasoning iterations (default: 20)",
    )

    parser.add_argument(
        "--executor",
        "-e",
        choices=["tci", "internal"],
        default="internal",
        help="Code execution mode: 'tci' for cloud, 'internal' for local Docker (default: internal)",
    )

    parser.add_argument(
        "--data-dir",
        "-d",
        help="Directory containing data files to upload. If not specified, will prompt to use current directory.",
    )

    parser.add_argument(
        "--session-id", "-s", help="Reuse an existing session ID (optional)"
    )

    parser.add_argument(
        "--write-report",
        "-w",
        action="store_true",
        help="Write a report to a file (default: False)",
    )

    args = parser.parse_args()

    # Validate inputs
    validate_executor(args.executor)

    if args.iterations < 1:
        console.print("[bold red]Error:[/bold red] Iterations must be at least 1")
        sys.exit(1)

    # Handle data directory
    data_dir = get_data_directory(args.data_dir)

    # Show configuration
    # Update args for display
    args.data_dir = (Path(data_dir).name if data_dir else "None (no files will be uploaded)")
    show_configuration(args)

    # Ask for confirmation
    if not Confirm.ask("\n[bold]Proceed with these settings?[/bold]", default=True):
        console.print("[yellow]Cancelled by user.[/yellow]")
        sys.exit(0)

    # Welcome message
    welcome_text = "🚀 Starting ReAct Data Science Agent"
    if data_dir:
        welcome_text += f"\n📁 Data from: {Path(data_dir).name}"
    welcome_text += f"\n🧠 Model: {args.model}"
    welcome_text += f"\n⚡ Executor: {args.executor.upper()}"

    welcome_panel = Panel(
        welcome_text,
        title="🤖 ReAct Data Science Agent",
        border_style="bold blue",
        expand=False,
    )
    console.print(welcome_panel)

    # Check API key for TCI mode
    if args.executor == "tci":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            console.print(
                "[bold red]Error:[/bold red] TOGETHER_API_KEY environment variable not set!"
            )
            console.print("[yellow]Please set your Together AI API key:[/yellow]")
            console.print("export TOGETHER_API_KEY='your-api-key-here'")
            sys.exit(1)

    # Create the agent
    try:
        agent = ReActDataScienceAgent(
            session_id=args.session_id,
            model=args.model,
            max_iterations=args.iterations,
            executor=args.executor,
            data_dir=data_dir,
        )

    except Exception as e:
        console.print(f"[bold red]Error creating agent:[/bold red] {str(e)}")
        sys.exit(1)

    # Interactive task input
    console.print("\n" + "=" * 80)
    console.print(
        "[bold green]Agent ready! Enter your data science task below.[/bold green]"
    )
    console.print("[dim]Type 'quit' or 'exit' to stop, or press Ctrl+C[/dim]")
    console.print("=" * 80 + "\n")

    try:
        while True:
            task = Prompt.ask(
                "[bold cyan]🎯 What would you like me to analyze?[/bold cyan]",
                default="",
            )

            if task.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break

            if not task.strip():
                console.print("[yellow]Please enter a task or 'quit' to exit.[/yellow]")
                continue

            # Run the analysis
            console.print("\n" + "=" * 80)
            result = agent.run(task)
            console.print("=" * 80 + "\n")

            if args.write_report:
                _write_report(user_input=task, result=result, history=agent.history, model=args.model)

            # Ask if user wants to continue
            if not Confirm.ask(
                "[bold]Would you like to run another analysis?[/bold]", default=True
            ):
                console.print("[green]Task completed successfully! 🎉[/green]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
