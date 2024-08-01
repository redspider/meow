"""
Meow is a simple cli interface to OpenAI LLMs
"""

import os
import sys
from datetime import datetime
from typing import Iterator, Tuple, Callable

import click
import openai
import pyperclip
import rich
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
)
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.live import Live

MEOW_STRING = "[bright_red]M[/bright_red][bright_yellow]e[/bright_yellow][bright_green]o[/bright_green][bright_blue]w[/bright_blue]"


@click.group()
def cli():
    pass


MODELS = ["gpt-4o", "gpt-4-turbo"]


def extract_code_blocks(s: str) -> list[str]:
    """
    Find all the code blocks in the provided markdown string (if any)
    """
    output: list[str] = []
    current = None
    lines = s.split("\n")
    for line in lines:
        if current is None:
            if line.startswith("```"):
                current = ""
        else:
            if line == "```":
                output.append(current)
                current = None
            else:
                current += line + "\n"

    return output


class MeowChat:
    """
    Chat service for OpenAI models
    """

    client: openai.OpenAI
    model: str
    history: list[
        ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam
    ]
    console: rich.console.Console

    def __init__(self, openai_api_key: str, model: str):
        self.client = openai.OpenAI(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1",
        )
        self.model = model
        self.console = rich.console.Console(record=True)
        self.reset_history()

    def get_system_prompt(self) -> str:
        return f"""You are an AI assistant called Meow, you use semi-formal language and are generally quite concise.
        Your user is an experienced computer programmer.

        The current date is: {datetime.now().strftime("%Y-%m-%d")}
        """

    def reset_history(self) -> None:
        self.history = [{"role": "system", "content": self.get_system_prompt()}]

    def command_model(self) -> None:
        """
        Toggle model
        """
        self.model = MODELS[(MODELS.index(self.model) + 1) % len(MODELS)]

    def command_copy(self) -> None:
        """
        Copy the last response to the clipboard
        """
        pyperclip.copy(self.history[-1]["content"])
        self.console.print("Copied last response to clipboard")

    def command_code_copy(self) -> None:
        """
        Copy the last code block to the clipboard
        """
        code_blocks = extract_code_blocks(self.history[-1]["content"])
        if not code_blocks:
            self.console.print("No code blocks found in last message")
        else:
            pyperclip.copy(code_blocks[-1])
            self.console.print("Copied last code block to clipboard")

    def command_dump(self) -> None:
        """
        Dump the conversation to screen raw with no formatting
        """
        for message in self.history:
            self.console.print(f'<{message["role"]}>\n{message["content"]}\n</{message["role"]}>', markup=False)

    def command_reset(self) -> None:
        """
        Start a new chat context
        """
        self.console.rule("Reset")
        self.reset_history()

    def command_quit(self) -> None:
        """
        Quit the application
        """
        sys.exit(0)

    def get_commands(self, with_long=False) -> Iterator[Tuple[str, str, str, Callable[[], None]]]:
        """
        Retrieve all the commands in this class
        """
        for method, fn in vars(self.__class__).items():
            if method.startswith("command_"):
                shortcut = "".join(c[0] for c in method.split("_")[1:])
                long = "_".join(c for c in method.split("_")[1:])
                yield shortcut, " ".join(method.split("_")[1:]).capitalize(), fn.__doc__.strip(), fn
                if with_long:
                    yield long, " ".join(method.split("_")[1:]).capitalize(), fn.__doc__.strip(), fn


    def run_command(self, command: str) -> None:
        """
        Run a command
        :param command:
        :return:
        """
        for shortcut, _, _, fn in self.get_commands(with_long=True):
            if shortcut == command:
                fn(self)
                return
        self.console.print(f"Unknown command: {command}")

    def command_help(self) -> None:
        """
        Show this help
        """
        self.console.print(f"[bold]{MEOW_STRING} commands[/bold]")

        for shortcut, name, help, _ in self.get_commands():
            self.console.print(f"[bold]\\{shortcut}[/bold] - {name}: {help}")

    @staticmethod
    def _get_prompt_toolkit_key_bindings() -> KeyBindings:
        """
        Get a configured keybindings so we get the right ENTER behaviour
        :return: KeyBindings object
        """
        kb = KeyBindings()

        @kb.add(Keys.Enter, eager=True)
        def _(event):
            buffer = event.app.current_buffer
            if buffer.document.text and buffer.document.text[0] == "\\":
                buffer.validate_and_handle()
            else:
                buffer.insert_text("\n")

        return kb

    def chat(self) -> None:
        """
        Start the chat
        """

        self.console.print(
            f"[bold]Welcome to {MEOW_STRING}. \\q to quit, \\h for help. meta-enter to submit (option-enter on OSX)[/bold]"
        )

        ptk_history = InMemoryHistory()
        ptk_key_bindings = self._get_prompt_toolkit_key_bindings()

        while True:
            try:
                # Prompt user for next input
                user_message = prompt(
                    "> ",
                    multiline=True,
                    style=Style.from_dict({"prompt": "bold yellow"}),
                    complete_while_typing=False,
                    bottom_toolbar=f" {self.model} model",
                    history=ptk_history,
                    key_bindings=ptk_key_bindings,
                ).strip()

                # If it starts with a \ it's a command, run the command then loop
                if user_message[0] == "\\":
                    self.run_command(user_message[1:])
                    continue

                # Otherwise it's a prompt, add it to the history
                self.history.append({"role": "user", "content": user_message})

                # Stream the completion
                with Live("", console=self.console, auto_refresh=True, refresh_per_second=10) as live:
                    aggregated_result = ""
                    for chunk in self.client.chat.completions.create(
                        model=self.model,
                        messages=self.history,
                        stream=True,
                    ):
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            aggregated_result += delta_content
                            live.update(aggregated_result)

                    # Add the result to the history
                    self.history.append({"role": "assistant", "content": aggregated_result})
                self.console.print("")
            except KeyboardInterrupt:
                self.console.print("[bright_red]Interrupted[/bright_red]")


@cli.command("chat")
@click.option("--model", "-m", default="gpt-4o", type=click.Choice(MODELS), help="The model to use")
def chat(model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        rich.print(
            "No OpenAI API key found in environment, please put it in the OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    mc = MeowChat(api_key, model)
    mc.chat()
