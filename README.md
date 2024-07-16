# Meow
A really basic CLI tool for talking to the OpenAI API in a chat form.

## Requirements
Python 3.11 or better, and poetry.

## Installation
```
cd meow
poetry install
```

## Running
To test it, run:

```
poetry run meow chat
```

You'll probably want a convenience script to activate the venv and run meow from anywhere. There are many ways to do 
this, but here's one. Put this in your `.bashrc` / `.zshrc` to allow you to run `meow` from anywhere to initiate a chat 
session:

```
alias meow='poetry -C ~/src/meow run meow chat'
```

To execute a prompt, you need to use meta-enter (option-enter on OSX, alt-enter on Linux) not just enter, this allows 
you to  write / paste multi-line prompts.

