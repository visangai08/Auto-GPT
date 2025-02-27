"""The application entry point.  Can be invoked by a CLI or any other front end application."""
import logging
import sys
from pathlib import Path

from colorama import Fore, Style

from autogpt.agent.agent import Agent
from autogpt.commands.command import CommandRegistry
from autogpt.config import Config, check_openai_api_key
from autogpt.configurator import create_config
from autogpt.logs import logger
from autogpt.memory import get_memory
from autogpt.plugins import scan_plugins
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT, construct_main_ai_config
from autogpt.utils import (
    get_current_git_branch,
    get_latest_bulletin,
    get_legal_warning,
    markdown_to_ansi_style,
)
from autogpt.workspace import Workspace
from scripts.install_plugin_deps import install_plugin_dependencies


def run_auto_gpt(
    continuous: bool,
    continuous_limit: int,
    ai_settings: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
    workspace_directory: str,
    install_plugin_deps: bool,
):
    # Configure logging before we do anything else.
    logger.set_level(logging.DEBUG if debug else logging.INFO)
    logger.speak_mode = speak

    cfg = Config()
    # TODO: fill in llm values here
    check_openai_api_key()
    create_config(
        continuous,
        continuous_limit,
        ai_settings,
        skip_reprompt,
        speak,
        debug,
        gpt3only,
        gpt4only,
        memory_type,
        browser_name,
        allow_downloads,
        skip_news,
    )

    if cfg.continuous_mode:
        for line in get_legal_warning().split("\n"):
            logger.warn(markdown_to_ansi_style(line), "법률:", Fore.RED)

    if not cfg.skip_news:
        motd, is_new_motd = get_latest_bulletin()
        if motd:
            motd = markdown_to_ansi_style(motd)
            for motd_line in motd.split("\n"):
                logger.info(motd_line, "뉴스:", Fore.GREEN)
            if is_new_motd and not cfg.chat_messages_enabled:
                input(
                    Fore.MAGENTA
                    + Style.BRIGHT
                    + "뉴스: 게시판이 업데이트되었습니다! 계속하려면 Enter 키를 누르세요..."
                    + Style.RESET_ALL
                )

        git_branch = get_current_git_branch()
        if git_branch and git_branch != "stable":
            logger.typewriter_log(
                "경고: ",
                Fore.RED,
                f"현재 `{git_branch}` 브랜치에서 실행 중입니다. "
                "- 지원되지 않는 브랜치입니다.",
            )
        if sys.version_info < (3, 10):
            logger.typewriter_log(
                "경고: ",
                Fore.RED,
                "이전 버전의 Python에서 실행 중입니다. "
                "일부 사용자는 이 버전에서 Auto-GPT의 "
                "특정 부분에서 문제를 발견했습니다. "
                "Python 3.10 이상으로 업그레이드하는 것이 좋습니다.",
            )

    if install_plugin_deps:
        install_plugin_dependencies()

    # TODO: have this directory live outside the repository (e.g. in a user's
    #   home directory) and have it come in as a command line argument or part of
    #   the env file.
    if workspace_directory is None:
        workspace_directory = Path(__file__).parent / "auto_gpt_workspace"
    else:
        workspace_directory = Path(workspace_directory)
    # TODO: pass in the ai_settings file and the env file and have them cloned into
    #   the workspace directory so we can bind them to the agent.
    workspace_directory = Workspace.make_workspace(workspace_directory)
    cfg.workspace_path = str(workspace_directory)

    # HACK: doing this here to collect some globals that depend on the workspace.
    file_logger_path = workspace_directory / "file_logger.txt"
    if not file_logger_path.exists():
        with file_logger_path.open(mode="w", encoding="utf-8") as f:
            f.write("파일 작업 로거 ")

    cfg.file_logger_path = str(file_logger_path)

    cfg.set_plugins(scan_plugins(cfg, cfg.debug_mode))
    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry()

    command_categories = [
        "autogpt.commands.analyze_code",
        "autogpt.commands.audio_text",
        "autogpt.commands.execute_code",
        "autogpt.commands.file_operations",
        "autogpt.commands.git_operations",
        "autogpt.commands.google_search",
        "autogpt.commands.image_gen",
        "autogpt.commands.improve_code",
        "autogpt.commands.twitter",
        "autogpt.commands.web_selenium",
        "autogpt.commands.write_tests",
        "autogpt.app",
        "autogpt.commands.task_statuses",
    ]
    logger.debug(
        f"다음 명령 카테고리는 비활성화됩니다: {cfg.disabled_command_categories}"
    )
    command_categories = [
        x for x in command_categories if x not in cfg.disabled_command_categories
    ]

    logger.debug(f"다음 명령 카테고리가 활성화됩니다: {command_categories}")

    for command_category in command_categories:
        command_registry.import_commands(command_category)

    ai_name = ""
    ai_config = construct_main_ai_config()
    ai_config.command_registry = command_registry
    if ai_config.ai_name:
        ai_name = ai_config.ai_name
    # print(prompt)
    # Initialize variables
    full_message_history = []
    next_action_count = 0

    # add chat plugins capable of report to logger
    if cfg.chat_messages_enabled:
        for plugin in cfg.plugins:
            if hasattr(plugin, "can_handle_report") and plugin.can_handle_report():
                logger.info(f"로거에 플러그인 로드: {plugin.__class__.__name__}")
                logger.chat_plugins.append(plugin)

    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(cfg, init=True)
    logger.typewriter_log(
        "사용 메모리 유형:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log("사용 브라우저:", Fore.GREEN, cfg.selenium_web_browser)
    system_prompt = ai_config.construct_full_prompt()
    if cfg.debug_mode:
        logger.typewriter_log("프롬프트:", Fore.GREEN, system_prompt)

    agent = Agent(
        ai_name=ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        command_registry=command_registry,
        config=ai_config,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace_directory,
    )
    agent.start_interaction_loop()
