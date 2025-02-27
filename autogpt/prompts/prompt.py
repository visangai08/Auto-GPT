from colorama import Fore

from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.llm import ApiManager
from autogpt.logs import logger
from autogpt.prompts.generator import PromptGenerator
from autogpt.setup import prompt_user
from autogpt.utils import clean_input

CFG = Config()

DEFAULT_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)


def build_default_prompt_generator() -> PromptGenerator:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    prompt_generator.add_constraint(
        "~4000 word limit for short term memory. Your short term memory is short, so"
        " immediately save important information to files."
    )
    prompt_generator.add_constraint(
        "If you are unsure how you previously did something or want to recall past"
        " events, thinking about similar events will help you remember."
    )
    prompt_generator.add_constraint("No user assistance")
    prompt_generator.add_constraint(
        "Exclusively use the commands listed below e.g. command_name"
    )

    # Add resources to the PromptGenerator object
    prompt_generator.add_resource(
        "Internet access for searches and information gathering."
    )
    prompt_generator.add_resource("Long Term memory management.")
    prompt_generator.add_resource(
        "GPT-3.5 powered Agents for delegation of simple tasks."
    )
    prompt_generator.add_resource("File output.")

    # Add performance evaluations to the PromptGenerator object
    prompt_generator.add_performance_evaluation(
        "Continuously review and analyze your actions to ensure you are performing to"
        " the best of your abilities."
    )
    prompt_generator.add_performance_evaluation(
        "Constructively self-criticize your big-picture behavior constantly."
    )
    prompt_generator.add_performance_evaluation(
        "Reflect on past decisions and strategies to refine your approach."
    )
    prompt_generator.add_performance_evaluation(
        "Every command has a cost, so be smart and efficient. Aim to complete tasks in"
        " the least number of steps."
    )
    prompt_generator.add_performance_evaluation("Write all code to a file.")
    return prompt_generator


def construct_main_ai_config() -> AIConfig:
    """Construct the prompt for the AI to respond to

    Returns:
        str: The prompt string
    """
    config = AIConfig.load(CFG.ai_settings_file)
    if CFG.skip_reprompt and config.ai_name:
        logger.typewriter_log("이름 :", Fore.GREEN, config.ai_name)
        logger.typewriter_log("역할 :", Fore.GREEN, config.ai_role)
        logger.typewriter_log("목표:", Fore.GREEN, f"{config.ai_goals}")
        logger.typewriter_log(
            "API 예산:",
            Fore.GREEN,
            "infinite" if config.api_budget <= 0 else f"${config.api_budget}",
        )
    elif config.ai_name:
        logger.typewriter_log(
            "돌아온 것을 환영합니다! ",
            Fore.GREEN,
            f"{config.ai_name}으로 돌아가시겠습니까?",
            speak_text=True,
        )
        should_continue = clean_input(
            f"""마지막 설정으로 계속 진행하시겠습니까?
이름:  {config.ai_name}
역할:  {config.ai_role}
목표: {config.ai_goals}
API 예산: {"infinite" if config.api_budget <= 0 else f"${config.api_budget}"}
계속 ({CFG.authorise_key}/{CFG.exit_key}): """
        )
        if should_continue.lower() == CFG.exit_key:
            config = AIConfig()

    if not config.ai_name:
        config = prompt_user()
        config.save(CFG.ai_settings_file)

    if CFG.restrict_to_workspace:
        logger.typewriter_log(
            "참고:이 에이전트가 만든 모든 파일/디렉토리는 다음 작업 공간에서 찾을 수 있습니다.:",
            Fore.YELLOW,
            f"{CFG.workspace_path}",
        )
    # set the total api budget
    api_manager = ApiManager()
    api_manager.set_total_budget(config.api_budget)

    # Agent Created, print message
    logger.typewriter_log(
        config.ai_name,
        Fore.LIGHTBLUE_EX,
        "은 다음과 같은 세부 정보로 생성되었습니다:",
        speak_text=True,
    )

    # Print the ai config details
    # Name
    logger.typewriter_log("이름:", Fore.GREEN, config.ai_name, speak_text=False)
    # Role
    logger.typewriter_log("역할:", Fore.GREEN, config.ai_role, speak_text=False)
    # Goals
    logger.typewriter_log("목표:", Fore.GREEN, "", speak_text=False)
    for goal in config.ai_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)

    return config
