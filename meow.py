"""
Meow is a simple cli interface to OpenAI LLMs
"""
from datetime import datetime

import click
import keyring
import openai
import pyperclip
import rich
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.live import Live


@click.group()
def cli():
    pass

MODELS = [
    'gpt-4o',
    'gpt-4-turbo'
]

SYSTEM_PROMPT = f"""You are an AI assistant, you use semi-formal language and are generally quite concise.
Your user is an experienced computer programmer.

The current date is: {datetime.now().strftime("%Y-%m-%d")}
"""

def get_api_key() -> str:
    """
    Obtain and return the OpenAI API key
    """
    api_key = keyring.get_password("meow", "openai_api_key")
    if not api_key:
        rich.print("No OpenAI API key found in keyring")
        api_key = rich.console.Console().input("Please paste your OpenAI API key: ", password=True)
        keyring.set_password("meow", "openai_api_key", api_key)
    return api_key

def extract_code_blocks(s: str) -> list[str]:
    """
    Find all the code blocks in the provided markdown string (if any)
    """
    output: list[str] = []
    current = None
    lines = s.split("\n")
    for l in lines:
        if current is None:
            if l.startswith("```"):
                current = ""
        else:
            if l == "```":
                output.append(current)
                current = None
            else:
                current += l + "\n"

    return output

@cli.command("chat")
@click.option("--model", "-m", default="gpt-4o", type=click.Choice(MODELS), help="The model to use")
def chat(
        model: str
):
    openai_client = openai.OpenAI(
        api_key=get_api_key(),
        base_url="https://api.openai.com/v1",
    )

    history = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    console = rich.console.Console(record=True)

    console.print(
        "[bold]Welcome to Meow. \\q to quit, \\h for help. meta-enter to submit (option-enter on OSX)[/bold]"
    )

    def toolbar():
        return f" {model} model"

    ptk_key_bindings = KeyBindings()
    ptk_history = InMemoryHistory()
    @ptk_key_bindings.add(Keys.Enter, eager=True)
    def _(event):
        """
        This just detects when text starts with a `\\` and the user preses enter (avoiding multiline mode in this special
        case)
        """
        buffer = event.app.current_buffer
        if buffer.document.text and buffer.document.text[0] == "\\":
            buffer.validate_and_handle()
        else:
            event.app.current_buffer.insert_text("\n")

    while True:
        try:
            user_message = prompt(
                f"> ",
                multiline=True,
                style=Style.from_dict({"prompt": "bold yellow"}),
                complete_while_typing=False,
                bottom_toolbar=toolbar,
                key_bindings=ptk_key_bindings,
                history=ptk_history,
            ).strip()
            if user_message == "\\q":
                break
            elif user_message == "\\r":
                history = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ]
                console.rule("Reset")
                continue
            elif user_message == "\\h":
                console.print(
                    """[bold]Meow commands[/bold]
[bold]\\h[/bold] Help
[bold]\\c[/bold] Copy the last response to the clipboard
[bold]\\cc[/bold] Copy the last code block to the clipboard
[bold]\\r[/bold] Reset
[bold]\\m[/bold] Switch model
[bold]\\q[/bold] Quit
"""
                )
                continue
            elif user_message == "\\m":
                # rotate to the next model in the MODELS list
                model = MODELS[(MODELS.index(model) + 1) % len(MODELS)]
                continue
            elif user_message == "\c":
                pyperclip.copy(history[-1]["content"])
                console.print("Copied last response to clipboard")
                continue
            elif user_message == "\\cc":
                code_blocks = extract_code_blocks(history[-1]["content"])
                if not code_blocks:
                    console.print("No code blocks found in last message")
                    continue
                pyperclip.copy(code_blocks[-1])
                console.print("Copied last code block to clipboard")
                continue

            history.append({
                "role": "user",
                "content": user_message
            })

            with Live("", console=console, auto_refresh=True, refresh_per_second=10) as live:
                aggregated_result = ""
                for chunk in openai_client.chat.completions.create(
                    model=model,
                    messages=history,
                    stream=True,
                ):
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        aggregated_result += delta_content
                        live.update(aggregated_result)

                history.append({
                    "role": "assistant",
                    "content": aggregated_result
                })
            console.print("")
        except KeyboardInterrupt:
            rich.print("[bright_red]Interrupted[/bright_red]")



