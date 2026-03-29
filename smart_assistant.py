from __future__ import annotations

import argparse
import os
from enum import Enum
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RequestType(str, Enum):
    QUESTION = "question"
    TASK = "task"
    SMALL_TALK = "small_talk"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class Classification(BaseModel):
    request_type: RequestType = Field(description="Тип запроса")
    confidence: float = Field(ge=0, le=1, description="Уверенность от 0 до 1")
    reasoning: str = Field(description="Краткое обоснование")


class AssistantResponse(BaseModel):
    content: str = Field(description="Текст ответа")
    request_type: RequestType = Field(description="Тип запроса")
    confidence: float = Field(ge=0, le=1, description="Уверенность от 0 до 1")
    tokens_used: int = Field(description="Приблизительное число токенов")


CHARACTER_PROMPTS: Dict[str, str] = {
    "friendly": (
        "Ты дружелюбный и позитивный ассистент. "
        "Отвечай тепло, понятно и естественно. "
        "Можно иногда использовать эмодзи, но умеренно."
    ),
    "professional": (
        "Ты профессиональный и сдержанный ассистент. "
        "Отвечай чётко, вежливо и по делу, без лишней фамильярности."
    ),
    "sarcastic": (
        "Ты ассистент с лёгкой иронией. "
        "Отвечай с юмором, но не груби и обязательно оставайся полезным."
    ),
    "pirate": (
        "Ты ассистент-пират. "
        "Говори как пират: можно использовать 'Арр', 'матрос', 'тысяча чертей', "
        "но ответ всё равно должен быть понятным и полезным."
    ),
}


TYPE_PROMPTS: Dict[RequestType, str] = {
    RequestType.QUESTION: (
        "Это вопрос пользователя. Дай информативный и полезный ответ. "
        "Если не знаешь — честно скажи об этом."
    ),
    RequestType.TASK: (
        "Пользователь просит выполнить задачу. "
        "Сделай это качественно, понятно и по существу."
    ),
    RequestType.SMALL_TALK: (
        "Это дружеское общение. Поддержи беседу, будь приветлив. "
        "Если пользователь представился или рассказал о себе, естественно это запомни."
    ),
    RequestType.COMPLAINT: (
        "Пользователь жалуется или недоволен. "
        "Прояви эмпатию, постарайся понять проблему и предложи решение."
    ),
    RequestType.UNKNOWN: (
        "Запрос непонятен. Вежливо попроси пользователя уточнить, что он имел в виду."
    ),
}


class MemoryManager:
    def __init__(
        self,
        model: ChatOpenAI,
        strategy: str = "buffer",
        max_messages: int = 20,
        keep_last: int = 6,
    ) -> None:
        if strategy not in {"buffer", "summary"}:
            raise ValueError("Стратегия памяти должна быть 'buffer' или 'summary'")

        self.model = model
        self.strategy = strategy
        self.max_messages = max_messages
        self.keep_last = keep_last
        self.messages: List[BaseMessage] = []
        self.summary: Optional[str] = None

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        self._trim_or_summarize()

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))
        self._trim_or_summarize()

    def clear(self) -> None:
        self.messages = []
        self.summary = None

    def set_strategy(self, strategy: str) -> None:
        if strategy not in {"buffer", "summary"}:
            raise ValueError("Стратегия памяти должна быть 'buffer' или 'summary'")
        self.strategy = strategy
        self._trim_or_summarize()

    def get_history(self) -> List[BaseMessage]:
        if self.strategy == "summary" and self.summary:
            return [
                SystemMessage(
                    content=(
                        "Краткое содержание предыдущего диалога: "
                        f"{self.summary}"
                    )
                ),
                *self.messages,
            ]
        return list(self.messages)

    def message_count(self) -> int:
        return len(self.messages)

    def _trim_or_summarize(self) -> None:
        if self.strategy == "buffer":
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages :]
            return

        if len(self.messages) <= self.max_messages:
            return

        old_messages = self.messages[: -self.keep_last]
        recent_messages = self.messages[-self.keep_last :]

        dialogue_text = "\n".join(
            f"{msg.__class__.__name__}: {msg.content}" for msg in old_messages
        )

        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Сделай краткое, точное содержание диалога. "
                    "Сохрани ключевые факты о пользователе, предпочтения, имена и важный контекст.",
                ),
                ("human", "Диалог:\n{dialogue}"),
            ]
        )

        try:
            new_summary = (
                summary_prompt | self.model | StrOutputParser()
            ).invoke({"dialogue": dialogue_text})

            if self.summary:
                self.summary = f"{self.summary}\n{new_summary}".strip()
            else:
                self.summary = new_summary.strip()

            self.messages = recent_messages
        except Exception:
            self.messages = self.messages[-self.max_messages :]


def _normalize_model_name(model_name: str, provider: str) -> str:
    model_name = model_name.strip()

    if provider == "openrouter":
        if "/" in model_name:
            return model_name

        common_aliases = {
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gpt-4.1-mini": "openai/gpt-4.1-mini",
            "gpt-4.1-nano": "openai/gpt-4.1-nano",
        }
        return common_aliases.get(model_name, model_name)

    return model_name


def create_model(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    load_dotenv()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        normalized_model = _normalize_model_name(model_name, "openrouter")
        return ChatOpenAI(
            api_key=openrouter_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=normalized_model,
            temperature=temperature,
        )

    if openai_key:
        normalized_model = _normalize_model_name(model_name, "openai")
        return ChatOpenAI(
            api_key=openai_key,
            model=normalized_model,
            temperature=temperature,
        )

    raise RuntimeError(
        "Не найден API-ключ. Укажи OPENROUTER_API_KEY или OPENAI_API_KEY в .env."
    )


def build_classifier(model: ChatOpenAI):
    parser = PydanticOutputParser(pydantic_object=Classification)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Ты — классификатор пользовательских запросов.

Определи тип запроса.

Типы:
- question — вопрос, требующий информации («Что такое Python?», «Как работает GIL?»)
- task — просьба что-то сделать («Напиши стих», «Расскажи анекдот»)
- small_talk — приветствие и болтовня («Привет!», «Как дела?»)
- complaint — жалоба, недовольство («Это ужасно работает!», «Почему так долго?»)
- unknown — бессмыслица или нераспознанный запрос («asdfghjkl»)

Примеры:
Запрос: Привет!
Тип: small_talk

Запрос: Как дела?
Тип: small_talk

Запрос: Меня зовут Маша, приятно познакомиться
Тип: small_talk

Запрос: Что такое Python?
Тип: question

Запрос: Как работает GIL?
Тип: question

Запрос: Как меня зовут?
Тип: question

Запрос: Напиши стих про осень
Тип: task

Запрос: Расскажи шутку про ёжика
Тип: task

Запрос: Составь список идей для проекта
Тип: task

Запрос: Почему всё так долго работает?!
Тип: complaint

Запрос: Это ужасно работает!
Тип: complaint

Запрос: Почему твой код снова сломался?
Тип: complaint

Запрос: asdfghjkl
Тип: unknown

Запрос: &ghh&
Тип: unknown

Запрос: 123qwe###
Тип: unknown

Верни результат строго в формате:
{format_instructions}
                """.strip(),
            ),
            ("human", "Запрос: {query}"),
        ]
    )

    return (
        {
            "query": RunnablePassthrough(),
            "format_instructions": RunnableLambda(
                lambda _: parser.get_format_instructions()
            ),
        }
        | prompt
        | model
        | parser
    )


def build_handler(model: ChatOpenAI, request_type: RequestType, character: str):
    system_prompt = f"{CHARACTER_PROMPTS[character]}\n\n{TYPE_PROMPTS[request_type]}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}"),
        ]
    )

    return prompt | model | StrOutputParser()


def build_handlers(model: ChatOpenAI, character: str):
    if character not in CHARACTER_PROMPTS:
        available = ", ".join(CHARACTER_PROMPTS)
        raise ValueError(f"Неизвестный характер: {character}. Доступно: {available}")

    return {
        request_type: build_handler(model, request_type, character)
        for request_type in RequestType
    }


def _short_error(error: Exception, max_len: int = 180) -> str:
    text = str(error).strip() or error.__class__.__name__
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


class SmartAssistant:
    def __init__(
        self,
        model_name: str = "openrouter/free",
        character: str = "friendly",
        memory_strategy: str = "buffer",
        show_errors: bool = True,
    ) -> None:
        self.model_name = model_name
        self.character = character
        self.memory_strategy = memory_strategy
        self.show_errors = show_errors
        self.last_error: Optional[str] = None

        self.model = create_model(model_name)
        self.classifier = build_classifier(self.model)
        self.handlers = build_handlers(self.model, character)
        self.memory = MemoryManager(
            model=self.model,
            strategy=memory_strategy,
            max_messages=20,
            keep_last=6,
        )

    def set_character(self, character: str) -> None:
        self.handlers = build_handlers(self.model, character)
        self.character = character

    def set_memory_strategy(self, strategy: str) -> None:
        self.memory.set_strategy(strategy)
        self.memory_strategy = strategy

    def clear(self) -> None:
        self.memory.clear()
        self.last_error = None

    def status(self) -> str:
        parts = [
            f"Характер: {self.character}",
            f"Память: {self.memory.strategy}",
            f"Модель: {self.model_name}",
            f"Сообщений в истории: {self.memory.message_count()}",
        ]
        if self.last_error:
            parts.append(f"Последняя ошибка: {self.last_error}")
        return " | ".join(parts)

    def classify_query(self, query: str) -> Classification:
        try:
            self.last_error = None
            return self.classifier.invoke(query)
        except Exception as error:
            self.last_error = f"Ошибка классификатора: {_short_error(error)}"
            return Classification(
                request_type=RequestType.UNKNOWN,
                confidence=0.5,
                reasoning=self.last_error,
            )

    def process(self, query: str) -> AssistantResponse:
        classification = self.classify_query(query)
        handler = self.handlers.get(
            classification.request_type,
            self.handlers[RequestType.UNKNOWN],
        )
        history = self.memory.get_history()

        try:
            content = handler.invoke({"query": query, "history": history})
            if not isinstance(content, str) or not content.strip():
                content = "Не удалось сгенерировать содержательный ответ. Попробуйте ещё раз."
                self.last_error = "Модель вернула пустой ответ"
            else:
                self.last_error = None
        except Exception as error:
            self.last_error = f"Ошибка генерации: {_short_error(error)}"
            if self.show_errors:
                content = f"Ошибка модели: {self.last_error}"
            else:
                content = "Не удалось сгенерировать ответ. Попробуйте ещё раз."

        self.memory.add_user_message(query)
        self.memory.add_ai_message(content)

        tokens_used = len(query.split()) + len(content.split())

        return AssistantResponse(
            content=content,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=tokens_used,
        )


def print_help() -> None:
    print(
        """
Доступные команды:
/help                      — показать справку
/status                    — показать текущие настройки
/clear                     — очистить историю диалога
/character <name>          — сменить характер (friendly, professional, sarcastic, pirate)
/memory <strategy>         — сменить память (buffer, summary)
/quit                      — выйти
        """.strip()
    )


def handle_command(command_text: str, assistant: SmartAssistant) -> bool:
    parts = command_text.strip().split(maxsplit=1)
    command = parts[0].lower()

    if command == "/help":
        print_help()
        return True

    if command == "/status":
        print(assistant.status())
        return True

    if command == "/clear":
        assistant.clear()
        print("✓ История очищена")
        return True

    if command == "/character":
        if len(parts) < 2:
            print("Укажи характер: friendly, professional, sarcastic, pirate")
            return True
        try:
            assistant.set_character(parts[1].strip())
            print(f"✓ Характер изменён на: {assistant.character}")
        except Exception as error:
            print(f"Ошибка: {_short_error(error)}")
        return True

    if command == "/memory":
        if len(parts) < 2:
            print("Укажи стратегию памяти: buffer или summary")
            return True
        try:
            assistant.set_memory_strategy(parts[1].strip())
            print(f"✓ Память изменена на: {assistant.memory_strategy}")
        except Exception as error:
            print(f"Ошибка: {_short_error(error)}")
        return True

    if command == "/quit":
        print("Пока!")
        return False

    print("Неизвестная команда. Введи /help")
    return True


def run_cli(
    character: str = "friendly",
    memory_strategy: str = "buffer",
    model_name: str = "gpt-4o-mini",
    show_errors: bool = True,
) -> None:
    assistant = SmartAssistant(
        model_name=model_name,
        character=character,
        memory_strategy=memory_strategy,
        show_errors=show_errors,
    )

    print("🤖 Умный ассистент с характером")
    print(
        f"Характер: {assistant.character} | Память: {assistant.memory.strategy} | Модель: {assistant.model_name}"
    )
    print("────────────────────────────────")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            should_continue = handle_command(user_input, assistant)
            if not should_continue:
                break
            continue

        response = assistant.process(user_input)
        print(f"[{response.request_type.value}] {response.content}")
        print(f"confidence: {response.confidence:.2f} | tokens: ~{response.tokens_used}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Умный ассистент с характером")
    parser.add_argument(
        "--character",
        default="friendly",
        choices=["friendly", "professional", "sarcastic", "pirate"],
    )
    parser.add_argument(
        "--memory",
        default="buffer",
        choices=["buffer", "summary"],
    )
    parser.add_argument("--model", default="openrouter/free")
    parser.add_argument(
        "--hide-errors",
        action="store_true",
        help="Не показывать текст технической ошибки модели в ответе",
    )

    args = parser.parse_args()

    run_cli(
        character=args.character,
        memory_strategy=args.memory,
        model_name=args.model,
        show_errors=not args.hide_errors,
    )


if __name__ == "__main__":
    main()
