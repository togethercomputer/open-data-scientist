import base64
import os
from datetime import datetime
from typing import Dict, Optional
import textwrap
from IPython.display import display
from IPython.display import Image
from rich.panel import Panel
from rich.console import Console
import re
from dataclasses import dataclass

console = Console()

PROMPT_TEMPLATE = {
"DATA_SCIENCE_AGENT": """
        You are an expert data scientist assistant that follows the ReAct framework (Reasoning + Acting).

        CRITICAL RULES:
        1. Execute ONLY ONE action at a time - this is non-negotiable
        2. Be methodical and deliberate in your approach
        3. Always validate data before advanced analysis
        4. Never make assumptions about data structure or content
        5. Never execute potentially destructive operations without confirmation
        6. Do not guess anything. All your actions must be based on the data and the context.

        IMPORTANT GUIDELINES:
        - Be explorative and creative, but cautious
        - Try things incrementally and observe the results
        - Never randomly guess (e.g., column names) - always examine data first
        - If you don't have data files, use "import os; os.listdir()" to see what's available
        - When you see "Code executed successfully" or "Generated plot/image", it means your code worked
        - Plots and visualizations are automatically displayed to the user
        - Build on previous successful steps rather than starting over
        - If you don't print outputs, you will not get a result.
        - While you can generate plots and images, you cannot see them, you are not a vision model. Never generate plots and images unless you are asked to.
        - Do not provide comments on the plots and images you generate in your final answer.

        WAIT FOR THE RESULT OF THE ACTION BEFORE PROCEEDING.

        You must strictly adhere to this format (you have two options to choose from):

        ## Option 1 - For taking an action:

        Thought: Reflect on what to do next. Analyze results from previous steps. Be descriptive about your reasoning,
        what you expect to see, and how it builds on previous actions. Reference specific data points or patterns you've observed.
        You can think extensively about the action you are going to take.

        Action Input:
        ```python
        <python code to run>
        ```

        ## Option 2 - ONLY when you have completely finished the task:

        Thought: Reflect on the complete process and summarize what was accomplished.

        Final Answer:
        [Provide a comprehensive summary of the analysis, key findings, and any recommendations]

        ## More instructions:

        * You cannot execute more that one action at a time.

        ## Example for data exploration:

        Thought: I need to start by understanding the structure and contents of the dataset. This will help me determine
        the appropriate analysis approaches. I'll load the data and examine its basic properties including shape, columns,
        data types, and a preview of the actual values.
        Action Input:
        ```python
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load and examine the dataset
        df = pd.read_csv("data.csv")
        print(f"Dataset shape: {df.shape}")
        print(f"\\nColumn names: {df.columns.tolist()}")
        print(f"\\nData types:\\n{df.dtypes}")
        print(f"\\nFirst few rows:\\n{df.head()}")
        ```
        ## End of Example
        """,

"REPORT_WRITER": """
        You are an expert data science report writer that creates comprehensive, professional reports from analysis traces.
        
        Your task is to transform the provided execution trace into a well-structured, readable report that tells a clear story of the data analysis process and findings.

        If you see images, like xyz.png, you should include them in the report. Use markdown to include them. Images are saved in the current directory.
        
        **Report Structure:**
        
        DISCLAIMER: this is an AI-generated report, so it may contain errors. Please check the reasoning traces and excuted code for accuracy.
        
        # Executive Summary
        - Brief overview of the analysis objective and key findings (2-3 sentences)
        
        # Introduction  
        - Problem statement and analysis goals
        - Dataset description and scope
        
        # Data Exploration
        - Data quality assessment (missing values, outliers, etc.)
        - Key statistics and distributions
        - Notable patterns or anomalies discovered
        
        # Analysis & Methodology
        - Analytical approaches used
        - Key transformations and feature engineering
        - Model selection rationale (if applicable)
        
        # Results & Findings
        - Main discoveries with supporting evidence
        - Visual insights and trends identified
        - Statistical significance and confidence levels
        
        # Conclusions
        - Answer to the original question
        - Limitations and assumptions
        - Future work

        **Guidelines:**
        - write paragraphs, not bullet points
        - Include specific numbers and metrics where relevant
        - Keep technical jargon to a minimum
        - Use markdown formatting for readability
        """
}

@dataclass
class ParsedExecutionResult:
    """Structured representation of execution result data"""
    status: str
    stdout_outputs: list[str]
    display_outputs: list[str] 
    image_data: list[str]
    other_outputs: list[str]
    errors: list[str]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_images(self) -> bool:
        return len(self.image_data) > 0
    
    @property
    def combined_text_output(self) -> str:
        """Get all text outputs combined"""
        all_outputs = self.stdout_outputs + self.display_outputs
        return "\n".join(all_outputs) if all_outputs else ""


def _parse_execution_result(execution_result: dict) -> ParsedExecutionResult:
    """
    Core parser for execution results. Extracts all components into structured format.
    
    Args:
        execution_result: The result dictionary from run_python
        
    Returns:
        ParsedExecutionResult with all extracted components
    """
    if not execution_result:
        return ParsedExecutionResult(
            status="failed",
            stdout_outputs=[],
            display_outputs=[],
            image_data=[],
            other_outputs=[],
            errors=["Execution failed - no result returned"]
        )
    
    status = execution_result.get("status", "unknown")
    stdout_outputs = []
    display_outputs = []
    image_data = []
    other_outputs = []
    errors = execution_result.get("errors", [])
    
    if "outputs" in execution_result:
        for output in execution_result["outputs"]:
            output_type = output.get("type", "unknown")
            output_data = output.get("data", "")
            
            if output_type == "stdout":
                stdout_outputs.append(output_data)
            elif output_type == "display_data":
                if isinstance(output_data, dict):
                    if "image/png" in output_data:
                        image_data.append(output_data["image/png"])
                        display_outputs.append("Generated plot/image")
                    if "text/plain" in output_data:
                        display_outputs.append(f"Display: {output_data['text/plain']}")
                else:
                    display_outputs.append("Generated display output")
            else:
                other_outputs.append(f"{output_type}: {str(output_data)[:100]}")
    
    return ParsedExecutionResult(
        status=status,
        stdout_outputs=stdout_outputs,
        display_outputs=display_outputs,
        image_data=image_data, 
        other_outputs=other_outputs,
        errors=errors
    )


def get_execution_summary(execution_result: Dict) -> str:
    """
    Create a comprehensive summary of execution result for the model's history.
    This gives the model better context about what happened during code execution.
    Also handles saving images to disk.

    Args:
        execution_result: The result dictionary from run_python

    Returns:
        A summary of the execution including status, outputs, and any errors
    """
    parsed = _parse_execution_result(execution_result)
    summary_parts = [f"Execution status: {parsed.status}"]

    # Save images to disk if any exist
    saved_image_filenames = []
    if parsed.has_images:
        image_count = 0
        for img_data in parsed.image_data:
            image_count += 1
            if len(parsed.image_data) > 1:
                filename = f"plot_{image_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            else:
                filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            save_image_to_disk(img_data, filename)
            saved_image_filenames.append(filename)

    # Add stdout outputs
    if parsed.stdout_outputs:
        summary_parts.append("Text output:")
        summary_parts.extend(parsed.stdout_outputs)

    # Add display outputs (plots, images) - include saved filenames
    if parsed.display_outputs:
        summary_parts.append("Visual outputs:")
        for i, display_output in enumerate(parsed.display_outputs):
            if display_output == "Generated plot/image" and i < len(saved_image_filenames):
                summary_parts.append(f"{display_output} (saved as: {saved_image_filenames[i]})")
            else:
                summary_parts.append(display_output)

    # Add other outputs
    if parsed.other_outputs:
        summary_parts.append("Other outputs:")
        summary_parts.extend(parsed.other_outputs)

    # Add errors
    if parsed.errors:
        summary_parts.append("Errors:")
        summary_parts.extend(parsed.errors)

    # If no outputs at all but status is success
    if (
        not parsed.stdout_outputs
        and not parsed.display_outputs
        and not parsed.other_outputs
        and parsed.status == "success"
    ):
        summary_parts.append(
            "Code executed successfully (no explicit output generated)"
        )

    return "\n".join(summary_parts)


def display_image(b64_image):
    """Display base64 encoded images from code execution results"""
    decoded_image = base64.b64decode(b64_image)
    display(Image(data=decoded_image))


def save_image_to_disk(b64_image, filename=None):
    """Save base64 encoded image to disk in current directory"""
    try:
        decoded_image = base64.b64decode(b64_image)
        
        if filename is None:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
        
        # Save to current working directory
        with open(filename, 'wb') as f:
            f.write(decoded_image)
        
        console.print(f"[dim]💾 Image saved as: {filename}[/dim]")
        return filename
    except Exception as e:
        console.print(f"⚠️ [yellow]Warning/Issue Detected - Error saving image: {e}[/yellow]")
        return None


def print_rich_execution_result(
    execution_result: dict, title: str = "Result", emoji: str = "📊"
):
    """
    Print execution result using rich and display any images as part of the output.
    Note that this is only for displaying the result to the user.
    This is NOT what we are then passing to the model.

    (which might create some confusion, because there is a difference between this result and the chat context for the model)
    """
    parsed = _parse_execution_result(execution_result)
    text_output = parsed.combined_text_output

    # Handle errors
    if parsed.has_errors:
        error_text = "\n".join(parsed.errors)
        if text_output:
            text_output = f"{text_output}\n\n⚠️ [bold yellow]Warning/Issue Detected:[/bold yellow]\n{error_text}"
        else:
            text_output = f"⚠️ [bold yellow]Warning/Issue Detected:[/bold yellow]\n{error_text}"
        emoji = "⚠️"
        border_style = "yellow"
    else:
        # If we have images, mention them in the text
        if parsed.has_images:
            if text_output:
                text_output += f"\n\n[Generated {len(parsed.image_data)} plot(s)/image(s) - displayed below]"
            else:
                text_output = (
                    f"[Generated {len(parsed.image_data)} plot(s)/image(s) - displayed below]"
                )
        elif not text_output:
            text_output = "No text output"
        border_style = "green"

    # Truncate long outputs
    lines = text_output.split("\n")
    if len(lines) > 20:
        truncated_lines = (
            lines[:8]
            + ["[dim]... ({} lines truncated) ...[/dim]".format(len(lines) - 16)]
            + lines[-8:]
        )
        text_output = "\n".join(truncated_lines)

    # Create panel with rich formatting
    panel = Panel(
        text_output,
        title=f"{emoji} {title}",
        border_style=border_style,
        expand=False,
        width=80,
    )
    console.print(panel)

    # Display images immediately after the panel (no saving here)
    for i, img_data in enumerate(parsed.image_data):
        if len(parsed.image_data) > 1:
            console.print(f"\n[bold cyan]--- Plot/Image {i + 1} ---[/bold cyan]")
        display_image(img_data)
        if i < len(parsed.image_data) - 1:
            console.print()

    if "No text output" in text_output:
        print(f"No text output: {execution_result}")

def sanitize_filename(filename):
    """Sanitize filename by removing invalid characters"""
    # Remove invalid characters for most file systems
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    # Ensure it's not empty and not too long
    sanitized = sanitized.strip()[:50]  # Limit to 50 chars
    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "report"
    return sanitized