"""Set up the AI and its goals"""
import re

from colorama import Fore, Style
from jinja2 import Template

from autogpt import utils
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.llm import create_chat_completion
from autogpt.logs import logger
from autogpt.prompts.default_prompts import (
    DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC,
    DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC,
    DEFAULT_USER_DESIRE_PROMPT,
)

CFG = Config()


def prompt_user() -> AIConfig:
    """Prompt the user for input

    Returns:
        AIConfig: The AIConfig object tailored to the user's input
    """
    ai_name = ""
    ai_config = None

    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "run with '--help' for more information.",
        speak_text=True,
    )

    # Get user desire
    logger.typewriter_log(
        "Create an AI-Assistant:",
        Fore.GREEN,
        "input '--manual' to enter manual mode.",
        speak_text=True,
    )

    user_desire = utils.clean_input(
        f"{Fore.LIGHTBLUE_EX}I want Auto-GPT to{Style.RESET_ALL}: "
    )

    if user_desire == "":
        user_desire = DEFAULT_USER_DESIRE_PROMPT  # Default prompt

    # If user desire contains "--manual"
    if "--manual" in user_desire:
        logger.typewriter_log(
            "Manual Mode Selected",
            Fore.GREEN,
            speak_text=True,
        )
        return generate_aiconfig_manual()

    else:
        try:
            return generate_aiconfig_automatic(user_desire)
        except Exception as e:
            logger.typewriter_log(
                "Unable to automatically generate AI Config based on user desire.",
                Fore.RED,
                "Falling back to manual mode.",
                speak_text=True,
            )

            return generate_aiconfig_manual()


def generate_aiconfig_manual() -> AIConfig:
    """
    Interactively create an AI configuration by prompting the user to provide the name, role, and goals of the AI.

    This function guides the user through a series of prompts to collect the necessary information to create
    an AIConfig object. The user will be asked to provide a name and role for the AI, as well as up to five
    goals. If the user does not provide a value for any of the fields, default values will be used.

    Returns:
        AIConfig: An AIConfig object containing the user-defined or default AI name, role, and goals.
    """

    # Manual Setup Intro
    logger.typewriter_log(
        "인공지능 어시스턴트를 만듭니다:",
        Fore.GREEN,
        "아래에 AI의 이름과 역할을 입력하세요. 아무것도 입력하지 않으면 로드됩니다."
        " defaults.",
        speak_text=True,
    )

    # Get AI Name from User
    logger.typewriter_log(
        "AI 이름 지정: ", Fore.GREEN, "예: 'Entrepreneur-GPT'"
    )
    ai_name = utils.clean_input("AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    logger.typewriter_log(
        f"{ai_name} here!", Fore.LIGHTBLUE_EX, "I am at your service.", speak_text=True
    )

    # Get AI Role from User
    logger.typewriter_log(
        "AI의 역할을 설명하세요.",
        Fore.GREEN,
        "예를 들어, '자율적으로 비즈니스를 개발하고 운영하도록 설계된 인공지능은"
        "순자산을 늘리는 것이 유일한 목표입니다."
    )
    ai_role = utils.clean_input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "순자산 증대를 유일한 목표로 비즈니스를 자율적으로 개발하고 운영하도록 설계된 AI입니다."

    # Enter up to 5 goals for the AI
    logger.typewriter_log(
        "인공지능에 대해 최대 5개의 목표를 입력하세요: ",
        Fore.GREEN,
        "예를 들어 예: \n순자산 증가, 트위터 계정 성장, 여러 비즈니스를 자율적으로 개발 및 관리",
    )
    logger.info("Enter nothing to load defaults, enter nothing when finished.")
    ai_goals = []
    for i in range(5):
        ai_goal = utils.clean_input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if not ai_goals:
        ai_goals = [
            "순자산 증가",
            "트위터 계정 키우기",
            "여러 비즈니스를 자율적으로 개발 및 관리",
        ]

    # Get API Budget from User
    logger.typewriter_log(
        "API 호출에 대한 예산을 입력합니다: ",
        Fore.GREEN,
        "예: $1.50",
    )
    logger.info("금액 제한 없이 AI가 실행되도록 하려면 아무것도 입력하지 마세요.")
    api_budget_input = utils.clean_input(
        f"{Fore.LIGHTBLUE_EX}Budget{Style.RESET_ALL}: $"
    )
    if api_budget_input == "":
        api_budget = 0.0
    else:
        try:
            api_budget = float(api_budget_input.replace("$", ""))
        except ValueError:
            logger.typewriter_log(
                "예산 입력이 잘못되었습니다. 예산을 무제한으로 설정했습니다.", Fore.RED
            )
            api_budget = 0.0

    return AIConfig(ai_name, ai_role, ai_goals, api_budget)


def generate_aiconfig_automatic(user_prompt) -> AIConfig:
    """Generates an AIConfig object from the given string.

    Returns:
    AIConfig: The AIConfig object tailored to the user's input
    """

    system_prompt = DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC
    prompt_ai_config_automatic = Template(
        DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC
    ).render(user_prompt=user_prompt)
    # Call LLM with the string as user input
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt_ai_config_automatic,
        },
    ]
    output = create_chat_completion(messages, CFG.fast_llm_model)

    # Debug LLM Output
    logger.debug(f"AI Config Generator Raw Output: {output}")

    # Parse the output
    ai_name = re.search(r"Name(?:\s*):(?:\s*)(.*)", output, re.IGNORECASE).group(1)
    ai_role = (
        re.search(
            r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
            output,
            re.IGNORECASE | re.DOTALL,
        )
        .group(1)
        .strip()
    )
    ai_goals = re.findall(r"(?<=\n)-\s*(.*)", output)
    api_budget = 0.0  # TODO: parse api budget using a regular expression

    return AIConfig(ai_name, ai_role, ai_goals, api_budget)
