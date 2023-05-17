import signal
import sys
from datetime import datetime

from colorama import Fore, Style

from autogpt.app import execute_command, get_command
from autogpt.config import Config
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import LLM_DEFAULT_RESPONSE_FORMAT, validate_json
from autogpt.llm import chat_with_ai, create_chat_completion, create_chat_message
from autogpt.llm.token_counter import count_string_tokens
from autogpt.log_cycle.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace


class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        full_message_history: The full message history.
        next_action_count: The number of actions to execute.
        system_prompt: The system prompt is the initial prompt that defines everything
          the AI needs to know to achieve its task successfully.
        Currently, the dynamic and customizable information in the system prompt are
          ai_name, description and goals.

        triggering_prompt: The last sentence the AI will see before answering.
            For Auto-GPT, this prompt is:
            Determine which next command to use, and respond using the format specified
              above:
            The triggering prompt is not part of the system prompt because between the
              system prompt and the triggering
            prompt we have contextual information that can distract the AI and make it
              forget that its goal is to find the next task to achieve.
            SYSTEM PROMPT
            CONTEXTUAL INFORMATION (memory, previous conversations, anything relevant)
            TRIGGERING PROMPT

        The triggering prompt reminds the AI about its short term meta task
        (defining the next task)
    """

    def __init__(
        self,
        ai_name,
        memory,
        full_message_history,
        next_action_count,
        command_registry,
        config,
        system_prompt,
        triggering_prompt,
        workspace_directory,
    ):
        cfg = Config()
        self.ai_name = ai_name
        self.memory = memory
        self.summary_memory = (
            "나는 창조되었습니다."  # Initial memory necessary to avoid hallucination
        )
        self.last_memory_index = 0
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.command_registry = command_registry
        self.config = config
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, cfg.restrict_to_workspace)
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cycle_count = 0
        self.log_cycle_handler = LogCycleHandler()

    def start_interaction_loop(self):
        # Interaction Loop
        cfg = Config()
        self.cycle_count = 0
        command_name = None
        arguments = None
        user_input = ""

        # Signal handler for interrupting y -N
        def signal_handler(signum, frame):
            if self.next_action_count == 0:
                sys.exit()
            else:
                print(
                    Fore.RED
                    + "인터럽트 신호 수신. 연속 명령 실행 중지."
                    + Style.RESET_ALL
                )
                self.next_action_count = 0

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            # Discontinue if continuous limit is reached
            self.cycle_count += 1
            self.log_cycle_handler.log_count_within_cycle = 0
            self.log_cycle_handler.log_cycle(
                self.config.ai_name,
                self.created_at,
                self.cycle_count,
                self.full_message_history,
                FULL_MESSAGE_HISTORY_FILE_NAME,
            )
            if (
                cfg.continuous_mode
                and cfg.continuous_limit > 0
                and self.cycle_count > cfg.continuous_limit
            ):
                logger.typewriter_log(
                    "연속 한도에 도달했습니다: ", Fore.YELLOW, f"{cfg.continuous_limit}"
                )
                break
            # Send message to AI, get response
            with Spinner("생각... "):
                assistant_reply = chat_with_ai(
                    self,
                    self.system_prompt,
                    self.triggering_prompt,
                    self.full_message_history,
                    self.memory,
                    cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)
            for plugin in cfg.plugins:
                if not plugin.can_handle_post_planning():
                    continue
                assistant_reply_json = plugin.post_planning(assistant_reply_json)

            # Print Assistant thoughts
            if assistant_reply_json != {}:
                validate_json(assistant_reply_json, LLM_DEFAULT_RESPONSE_FORMAT)
                # Get command name and arguments
                try:
                    print_assistant_thoughts(
                        self.ai_name, assistant_reply_json, cfg.speak_mode
                    )
                    command_name, arguments = get_command(assistant_reply_json)
                    if cfg.speak_mode:
                        say_text(f"{command_name}을 실행하고 싶습니다.")

                    arguments = self._resolve_pathlike_command_args(arguments)

                except Exception as e:
                    logger.error("오류: \n", str(e))
            self.log_cycle_handler.log_cycle(
                self.config.ai_name,
                self.created_at,
                self.cycle_count,
                assistant_reply_json,
                NEXT_ACTION_FILE_NAME,
            )

            logger.typewriter_log(
                "다음 작업: ",
                Fore.CYAN,
                f"명령 = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                f"인자 = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
            )

            if not cfg.continuous_mode and self.next_action_count == 0:
                # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                self.user_input = ""
                logger.info(
                    "명령을 승인하려면 'y', N개의 연속 명령을 실행하려면 'y -N', 자체 피드백 명령을 실행하려면 's', "
                    f"프로그램을 종료하려면 'n'을 입력하거나 {self.ai_name}...에 대한 피드백을 입력합니다..."
                )
                while True:
                    if cfg.chat_messages_enabled:
                        console_input = clean_input("응답을 기다리는 중입니다...")
                    else:
                        console_input = clean_input(
                            Fore.MAGENTA + "입력:" + Style.RESET_ALL
                        )
                    if console_input.lower().strip() == cfg.authorise_key:
                        user_input = "다음 명령 json 생성"
                        break
                    elif console_input.lower().strip() == "s":
                        logger.typewriter_log(
                            "-=-=-=-=-=-=-= 이제 에이전트가 생각, 추론, 계획 및 비판을 검증합니다. -=-=-=-=-=-=-=",
                            Fore.GREEN,
                            "",
                        )
                        thoughts = assistant_reply_json.get("생각", {})
                        self_feedback_resp = self.get_self_feedback(
                            thoughts, cfg.fast_llm_model
                        )
                        logger.typewriter_log(
                            f"자체 피드백: {self_feedback_resp}",
                            Fore.YELLOW,
                            "",
                        )
                        user_input = self_feedback_resp
                        command_name = "self_feedback"
                        break
                    elif console_input.lower().strip() == "":
                        logger.warn("입력 형식이 잘못되었습니다.")
                        continue
                    elif console_input.lower().startswith(f"{cfg.authorise_key} -"):
                        try:
                            self.next_action_count = abs(
                                int(console_input.split(" ")[1])
                            )
                            user_input = "다음 명령 json 생성"
                        except ValueError:
                            logger.warn(
                                "입력 형식이 잘못되었습니다. 'y -n'을 입력하세요."
                                " 여기서 n은 연속 작업의 수입니다."
                            )
                            continue
                        break
                    elif console_input.lower() == cfg.exit_key:
                        user_input = "EXIT"
                        break
                    else:
                        user_input = console_input
                        command_name = "human_feedback"
                        self.log_cycle_handler.log_cycle(
                            self.config.ai_name,
                            self.created_at,
                            self.cycle_count,
                            user_input,
                            USER_INPUT_FILE_NAME,
                        )
                        break

                if user_input == "다음 명령 json 생성":
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= 사용자가 승인한 명령 -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "",
                    )
                elif user_input == "종료":
                    logger.info("종료중...")
                    break
            else:
                # Print authorized commands left value
                logger.typewriter_log(
                    f"{Fore.CYAN}승인된 명령이 남았습니다: {Style.RESET_ALL}{self.next_action_count}"
                )

            # Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                result = (
                    f"{command_name} 명령에서 다음 오류가 발생했습니다: {arguments}"
                )
            elif command_name == "human_feedback":
                result = f"입력 피드백: {user_input}"
            elif command_name == "self_feedback":
                result = f"자체 피드백: {user_input}"
            else:
                for plugin in cfg.plugins:
                    if not plugin.can_handle_pre_command():
                        continue
                    command_name, arguments = plugin.pre_command(
                        command_name, arguments
                    )
                command_result = execute_command(
                    self.command_registry,
                    command_name,
                    arguments,
                    self.config.prompt_generator,
                )
                result = f"{command_name} 명령이 반환되었습니다: " f"{command_result}"

                result_tlength = count_string_tokens(
                    str(command_result), cfg.fast_llm_model
                )
                memory_tlength = count_string_tokens(
                    str(self.summary_memory), cfg.fast_llm_model
                )
                if result_tlength + memory_tlength + 600 > cfg.fast_token_limit:
                    result = f"실패: {command_name} 명령이 너무 많은 출력을 반환했습니다. \
                        동일한 인수로 이 명령을 다시 실행하지 마십시오."

                for plugin in cfg.plugins:
                    if not plugin.can_handle_post_command():
                        continue
                    result = plugin.post_command(command_name, result)
                if self.next_action_count > 0:
                    self.next_action_count -= 1

            # Check if there's a result from the command append it to the message
            # history
            if result is not None:
                self.full_message_history.append(create_chat_message("system", result))
                logger.typewriter_log("시스템: ", Fore.YELLOW, result)
            else:
                self.full_message_history.append(
                    create_chat_message("시스템", "명령을 실행할 수 없습니다")
                )
                logger.typewriter_log(
                    "시스템: ", Fore.YELLOW, "명령을 실행할 수 없습니다."
                )

    def _resolve_pathlike_command_args(self, command_args):
        if "directory" in command_args and command_args["directory"] in {"", "/"}:
            command_args["directory"] = str(self.workspace.root)
        else:
            for pathlike in ["filename", "directory", "clone_path"]:
                if pathlike in command_args:
                    command_args[pathlike] = str(
                        self.workspace.get_path(command_args[pathlike])
                    )
        return command_args

    def get_self_feedback(self, thoughts: dict, llm_model: str) -> str:
        """Generates a feedback response based on the provided thoughts dictionary.
        This method takes in a dictionary of thoughts containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. It combines these elements into a single
        feedback message and uses the create_chat_completion() function to generate a
        response based on the input message.
        Args:
            thoughts (dict): A dictionary containing thought elements like reasoning,
            plan, thoughts, and criticism.
        Returns:
            str: A feedback response generated using the provided thoughts dictionary.
        """
        ai_role = self.config.ai_role

        feedback_prompt = f"아래는 {ai_role}의 역할을 맡은 인공지능 에이전트인 제가 보낸 메시지입니다. 인공지능 에이전트로서 저의 약간의 한계를 알고 있으면서 저의 사고 과정, 추론, 계획을 평가해 주시고 잠재적인 개선 사항을 간결한 문단으로 정리해 주세요. 제 역할에 맞지 않는 아이디어를 추가 또는 제거하고 그 이유를 설명하거나, 중요도에 따라 생각의 우선순위를 정하거나, 전반적인 사고 과정을 개선하는 것을 고려하세요."
        reasoning = thoughts.get("추론", "")
        plan = thoughts.get("계획", "")
        thought = thoughts.get("생각", "")
        feedback_thoughts = thought + reasoning + plan
        return create_chat_completion(
            [{"role": "user", "content": feedback_prompt + feedback_thoughts}],
            llm_model,
        )
