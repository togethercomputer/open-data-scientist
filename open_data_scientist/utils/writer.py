from open_data_scientist.utils.strings import sanitize_filename
from open_data_scientist.utils.strings import PROMPT_TEMPLATE
from together import Client

def _format_history(history):
    """Format history messages into a readable summary"""
    if not history:
        return "No previous conversation history."
    
    formatted_parts = []
    
    for i, message in enumerate(history, 1):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        formatted_parts.append(f"{i}. {role}: {content}")
    
    return "\n\n".join(formatted_parts)


def _write_report(user_input, result, history, model="deepseek-ai/DeepSeek-V3"):
    """Write the report to a file"""

    client = Client()

    formatted_history = _format_history(history)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": PROMPT_TEMPLATE["REPORT_WRITER"]}, {"role": "user", "content": f"Conversation History:\n{formatted_history}\n\nUser input: {user_input}\nFinal result: {result}"}],
        temperature=0.5,
        stream=False,
    )
    output_report : str = response.choices[0].message.content  # type: ignore

    report_name = f"{sanitize_filename(user_input)}.md"
    
    with open(report_name, "w") as f:
        f.write(output_report)
    