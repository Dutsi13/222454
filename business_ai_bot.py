import json
import logging
import mimetypes
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Deque, Dict, List, Optional, Set

from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Message, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)


CONFIG_PATH = Path("config.env")
STATE_PATH = Path("data/business_ai_state.json")
LOG_PATH = Path("logs/events.jsonl")
DOWNLOAD_DIR = Path(gettempdir()) / "telegram_business_ai_bot"
DEFAULT_PROMPT = (
    "Ты AI-ассистент для Telegram Business. Отвечай естественно, кратко и по делу. "
    "Не пиши, что ты бот или ИИ. Не придумывай факты, если чего-то не знаешь. "
    "Сохраняй спокойный человеческий стиль без излишней официальности."
)
STYLE_ANALYSIS_PROMPT = """
Ты анализируешь скриншоты переписки, чтобы понять манеру общения автора сообщений.

Задача:
1. Выдели только стиль общения, лексику, длину фраз, типичную пунктуацию, форму приветствий и прощаний.
2. Не копируй длинные фрагменты текста из скриншотов.
3. Не пересказывай приватные данные, контакты, реквизиты, usernames и прочие чувствительные детали.
4. Сформируй краткий профиль стиля, который потом можно использовать для генерации похожих ответов.
5. Если на скриншоте мало текста, так и скажи.

Верни ответ в таком виде:
STYLE_SUMMARY:
...

DO:
- ...
- ...

AVOID:
- ...
- ...
""".strip()


load_dotenv(dotenv_path=CONFIG_PATH)


@dataclass
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    gemini_model: str
    vision_model: str
    log_level: str
    keywords: Set[str]
    admin_chat_id: Optional[int]
    default_prompt: str
    reply_to_private_messages: bool


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {
                "custom_prompt": "",
                "style_notes": [],
                "style_summary": "",
                "updated_at": utc_now_iso(),
            }

        try:
            with self.path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
        except (OSError, json.JSONDecodeError):
            payload = {}

        return {
            "custom_prompt": str(payload.get("custom_prompt", "")).strip(),
            "style_notes": list(payload.get("style_notes", [])),
            "style_summary": str(payload.get("style_summary", "")).strip(),
            "updated_at": str(payload.get("updated_at", utc_now_iso())),
        }

    def save(self) -> None:
        self.data["updated_at"] = utc_now_iso()
        with self.path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.data, file_obj, ensure_ascii=False, indent=2)

    def get_custom_prompt(self) -> str:
        return str(self.data.get("custom_prompt", "")).strip()

    def set_custom_prompt(self, value: str) -> None:
        self.data["custom_prompt"] = value.strip()
        self.save()

    def clear_custom_prompt(self) -> None:
        self.data["custom_prompt"] = ""
        self.save()

    def get_style_summary(self) -> str:
        return str(self.data.get("style_summary", "")).strip()

    def get_style_notes(self) -> List[str]:
        return [str(note).strip() for note in self.data.get("style_notes", []) if str(note).strip()]

    def append_style_note(self, note: str) -> None:
        cleaned = note.strip()
        if not cleaned:
            return
        notes = self.get_style_notes()
        notes.append(cleaned)
        self.data["style_notes"] = notes[-20:]
        self.data["style_summary"] = "\n\n".join(self.data["style_notes"])
        self.save()

    def clear_style(self) -> None:
        self.data["style_notes"] = []
        self.data["style_summary"] = ""
        self.save()


class GeminiAssistant:
    def __init__(self, api_key: str, text_model: str, vision_model: str, base_prompt: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.text_model = text_model
        self.vision_model = vision_model
        self.base_prompt = base_prompt

    def build_system_prompt(self, state: StateStore, chat_type: str) -> str:
        sections = [
            self.base_prompt,
            f"Тип текущего чата: {chat_type}.",
        ]

        custom_prompt = state.get_custom_prompt()
        if custom_prompt:
            sections.append("Дополнительные инструкции владельца:\n" + custom_prompt)

        style_summary = state.get_style_summary()
        if style_summary:
            sections.append(
                "Профиль стиля владельца, извлеченный из примеров переписки. "
                "Следуй ему мягко, без дословного копирования:\n" + style_summary
            )

        sections.append(
            "Пиши так, будто это обычный живой ответ человека в мессенджере. "
            "Если вопрос требует фактов и ты не уверен, отвечай осторожно и без выдумки."
        )
        return "\n\n".join(sections)

    def reply(self, state: StateStore, chat_type: str, history: List[Dict[str, str]], user_text: str) -> str:
        contents: List[types.Content] = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Ниже история диалога и новое сообщение. "
                            "Сгенерируй только ответ для отправки собеседнику.\n\n"
                            f"История:\n{format_history(history)}\n\n"
                            f"Новое сообщение:\n{user_text or 'Напиши короткое приветствие и уточни, чем помочь.'}"
                        )
                    ),
                ],
            )
        ]

        response = self.client.models.generate_content(
            model=self.text_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.build_system_prompt(state=state, chat_type=chat_type),
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=500,
            ),
        )
        return clean_model_text(response.text)

    def analyze_style_screenshot(self, image_bytes: bytes, mime_type: str, note: str = "") -> str:
        prompt = STYLE_ANALYSIS_PROMPT
        if note.strip():
            prompt += f"\n\nКомментарий владельца к этому скриншоту:\n{note.strip()}"

        response = self.client.models.generate_content(
            model=self.vision_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=600,
            ),
        )
        return clean_model_text(response.text)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_model_text(text: Optional[str]) -> str:
    cleaned = (text or "").strip()
    return cleaned or "Сейчас не получилось подготовить ответ, попробуй написать еще раз."


def load_settings() -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required in config.env")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required in config.env")

    keywords = {
        keyword.strip().lower()
        for keyword in os.getenv("KEYWORDS", "").split(",")
        if keyword.strip()
    }

    admin_chat_id_raw = os.getenv("ADMIN_CHAT_ID", "").strip()
    admin_chat_id = int(admin_chat_id_raw) if admin_chat_id_raw else None

    return Settings(
        telegram_bot_token=token,
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        vision_model=os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        keywords=keywords,
        admin_chat_id=admin_chat_id,
        default_prompt=os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT).strip() or DEFAULT_PROMPT,
        reply_to_private_messages=os.getenv("REPLY_TO_PRIVATE_MESSAGES", "true").strip().lower() in {"1", "true", "yes"},
    )


def format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "История пуста."
    rows = []
    for item in history[-12:]:
        role = item.get("role", "user")
        text = item.get("text", "").strip()
        rows.append(f"{role}: {text}")
    return "\n".join(rows)


def extract_text(message: Optional[Message]) -> str:
    if not message:
        return ""
    return (message.text or message.caption or "").strip()


def detect_message_content_type(message: Message) -> str:
    if message.text:
        return "text"
    if message.caption:
        if message.photo:
            return "photo_with_caption"
        if message.document:
            return "document_with_caption"
        return "caption_only"
    if message.photo:
        return "photo"
    if message.document:
        return "document"
    if message.voice:
        return "voice"
    if message.video:
        return "video"
    if message.video_note:
        return "video_note"
    if message.animation:
        return "animation"
    if message.audio:
        return "audio"
    if message.sticker:
        return "sticker"
    if message.contact:
        return "contact"
    if message.location:
        return "location"
    if message.venue:
        return "venue"
    if message.poll:
        return "poll"
    if message.story:
        return "story"
    if message.effective_attachment is not None:
        return type(message.effective_attachment).__name__.lower()
    return "unknown"


def build_user_input_from_message(message: Message) -> str:
    text = extract_text(message)
    if text:
        return text

    content_type = detect_message_content_type(message)
    descriptions = {
        "photo": "Пользователь прислал фото без подписи.",
        "document": "Пользователь прислал документ без подписи.",
        "voice": "Пользователь прислал голосовое сообщение без расшифровки.",
        "video": "Пользователь прислал видео без подписи.",
        "video_note": "Пользователь прислал кружок без текста.",
        "animation": "Пользователь прислал анимацию без подписи.",
        "audio": "Пользователь прислал аудио без подписи.",
        "sticker": "Пользователь прислал стикер.",
        "contact": "Пользователь прислал контакт.",
        "location": "Пользователь прислал геолокацию.",
        "venue": "Пользователь прислал место на карте.",
        "poll": "Пользователь прислал опрос.",
        "story": "Пользователь прислал story.",
        "unknown": "Пользователь прислал сообщение без доступного текста.",
    }
    return descriptions.get(content_type, f"Пользователь прислал сообщение типа {content_type} без текста.")


def message_to_dict(message: Message, update: Update, source: str) -> Dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "source": source,
        "update_id": update.update_id,
        "chat_id": message.chat_id,
        "chat_type": message.chat.type,
        "message_id": message.message_id,
        "from_user_id": message.from_user.id if message.from_user else None,
        "from_username": message.from_user.username if message.from_user else None,
        "business_connection_id": getattr(message, "business_connection_id", None),
        "text": message.text,
        "caption": message.caption,
        "content_type": detect_message_content_type(message),
        "has_attachment": message.effective_attachment is not None,
        "raw_message": message.to_dict(),
    }


def find_keyword_hits(text: str, keywords: Set[str]) -> List[str]:
    lowered = text.lower()
    return sorted([keyword for keyword in keywords if keyword in lowered])


def is_admin_message(message: Message, settings: Settings) -> bool:
    if settings.admin_chat_id is None:
        return message.chat.type == "private"
    return message.chat_id == settings.admin_chat_id


async def maybe_alert_admin(
    context: ContextTypes.DEFAULT_TYPE,
    admin_chat_id: Optional[int],
    text: str,
) -> None:
    if admin_chat_id is None:
        return
    await context.bot.send_message(chat_id=admin_chat_id, text=text)


async def reject_non_admin(message: Message, settings: Settings) -> bool:
    if is_admin_message(message, settings):
        return False
    await message.reply_text("Эта команда доступна только в админском чате.")
    return True


def get_history_store(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Deque[Dict[str, str]]]:
    return context.application.bot_data["history_store"]


def append_history(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    role: str,
    text: str,
) -> None:
    if not text.strip():
        return
    history_store = get_history_store(context)
    history_store[str(chat_id)].append({"role": role, "text": text.strip()})


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    await message.reply_text(
        "Команды:\n"
        "/setprompt - изменить текущий промпт\n"
        "/showprompt - показать активный промпт\n"
        "/resetprompt - сбросить дополнительный промпт\n"
        "/learnstyle - включить режим обучения по скриншотам\n"
        "/done - выключить режим обучения\n"
        "/showstyle - показать текущий профиль стиля\n"
        "/clearstyle - очистить стиль\n"
        "/status - показать текущие настройки"
    )


async def setprompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    if await reject_non_admin(message, settings):
        return

    new_prompt = " ".join(context.args).strip()
    if new_prompt:
        state.set_custom_prompt(new_prompt)
        await message.reply_text("Дополнительный промпт обновлен.")
        return

    context.user_data["awaiting_prompt"] = True
    await message.reply_text("Отправь следующим сообщением новый промпт целиком.")


async def showprompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    if await reject_non_admin(message, settings):
        return

    custom_prompt = state.get_custom_prompt() or "Не задан. Используется только базовый системный промпт."
    await message.reply_text(
        "Базовый промпт:\n"
        f"{settings.default_prompt}\n\n"
        "Дополнительный промпт:\n"
        f"{custom_prompt}"
    )


async def resetprompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    if await reject_non_admin(message, settings):
        return

    state.clear_custom_prompt()
    await message.reply_text("Дополнительный промпт очищен.")


async def learnstyle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    if await reject_non_admin(message, settings):
        return

    context.user_data["learning_style"] = True
    context.user_data["style_note"] = " ".join(context.args).strip()
    await message.reply_text(
        "Режим обучения по скриншотам включен.\n"
        "Теперь отправляй фото или изображения документом. Когда закончишь, отправь /done."
    )


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    if await reject_non_admin(message, settings):
        return

    context.user_data.pop("learning_style", None)
    context.user_data.pop("style_note", None)
    context.user_data.pop("awaiting_prompt", None)
    await message.reply_text("Режим ввода завершен.")


async def showstyle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    if await reject_non_admin(message, settings):
        return

    summary = state.get_style_summary() or "Стиль пока не обучен."
    await message.reply_text(summary[:3900])


async def clearstyle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    if await reject_non_admin(message, settings):
        return

    state.clear_style()
    await message.reply_text("Сохраненный профиль стиля очищен.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    if await reject_non_admin(message, settings):
        return

    await message.reply_text(
        "Текущий статус:\n"
        f"- text model: {settings.gemini_model}\n"
        f"- vision model: {settings.vision_model}\n"
        f"- custom prompt: {'yes' if state.get_custom_prompt() else 'no'}\n"
        f"- style notes: {len(state.get_style_notes())}\n"
        f"- reply to private messages: {settings.reply_to_private_messages}"
    )


async def process_prompt_input(message: Message, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not context.user_data.get("awaiting_prompt"):
        return False

    state: StateStore = context.application.bot_data["state"]
    state.set_custom_prompt(extract_text(message))
    context.user_data.pop("awaiting_prompt", None)
    await message.reply_text("Промпт сохранен.")
    return True


async def download_image_bytes(message: Message) -> Optional[tuple[bytes, str]]:
    telegram_file = None
    mime_type = ""

    if message.photo:
        telegram_file = await message.photo[-1].get_file()
        mime_type = "image/jpeg"
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        telegram_file = await message.document.get_file()
        mime_type = message.document.mime_type or ""
    else:
        return None

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DOWNLOAD_DIR / f"{message.chat_id}_{message.message_id}"
    await telegram_file.download_to_drive(custom_path=str(file_path))
    image_bytes = file_path.read_bytes()
    detected_mime = mime_type or mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    return image_bytes, detected_mime


async def process_style_image(message: Message, context: ContextTypes.DEFAULT_TYPE) -> bool:
    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    ai: GeminiAssistant = context.application.bot_data["ai"]

    if not context.user_data.get("learning_style"):
        return False
    if not is_admin_message(message, settings):
        return False

    image_payload = await download_image_bytes(message)
    if image_payload is None:
        await message.reply_text("Пришли именно скриншот: фото или изображение документом.")
        return True

    image_bytes, mime_type = image_payload
    note = context.user_data.get("style_note", "")
    status_message = await message.reply_text("Анализирую скриншот и обновляю профиль стиля...")

    try:
        style_note = ai.analyze_style_screenshot(image_bytes=image_bytes, mime_type=mime_type, note=note)
        state.append_style_note(style_note)
        await status_message.edit_text(
            "Стиль обновлен. Краткая выжимка из этого скриншота:\n\n"
            f"{style_note[:3500]}"
        )
    except Exception as exc:
        logging.exception("Style learning failed: %s", exc)
        await status_message.edit_text("Не получилось обработать скриншот. Проверь API ключ и модель Gemini.")

    return True


async def maybe_reply_with_ai(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    source: str,
) -> None:
    message = update.business_message if source == "business_message" else update.message
    if not message or not message.from_user:
        return

    if message.from_user.is_bot:
        return

    settings: Settings = context.application.bot_data["settings"]
    state: StateStore = context.application.bot_data["state"]
    ai: GeminiAssistant = context.application.bot_data["ai"]
    logs: JsonlLogger = context.application.bot_data["logs"]

    if source == "business_message" and settings.admin_chat_id and message.from_user.id == settings.admin_chat_id:
        return

    if source == "message" and not settings.reply_to_private_messages:
        return

    if source == "message" and message.chat.type != "private":
        return

    user_input = build_user_input_from_message(message)
    chat_id = message.chat_id
    append_history(context, chat_id, "user", user_input)

    try:
        history = list(get_history_store(context)[str(chat_id)])
        ai_answer = ai.reply(
            state=state,
            chat_type=message.chat.type,
            history=history,
            user_text=user_input,
        )
        await message.reply_text(ai_answer)
        append_history(context, chat_id, "assistant", ai_answer)
        logs.write(
            {
                "timestamp": utc_now_iso(),
                "event": "ai_reply",
                "source": source,
                "chat_id": chat_id,
                "message_id": message.message_id,
                "content_type": detect_message_content_type(message),
                "prompt_input": user_input,
                "answer": ai_answer,
            }
        )
    except Exception as exc:
        logging.exception("AI reply failed: %s", exc)
        logs.write(
            {
                "timestamp": utc_now_iso(),
                "event": "ai_reply_error",
                "source": source,
                "chat_id": chat_id,
                "message_id": message.message_id,
                "content_type": detect_message_content_type(message),
                "prompt_input": user_input,
                "error": repr(exc),
            }
        )
        fallback_text = (
            "Привет. Сообщение получил, но сейчас временно не могу нормально ответить. "
            "Напиши еще раз чуть позже или отправь вопрос текстом."
        )
        try:
            await message.reply_text(fallback_text)
            logs.write(
                {
                    "timestamp": utc_now_iso(),
                    "event": "ai_reply_fallback_sent",
                    "source": source,
                    "chat_id": chat_id,
                    "message_id": message.message_id,
                }
            )
        except Exception as send_exc:
            logging.exception("Fallback reply failed: %s", send_exc)
            logs.write(
                {
                    "timestamp": utc_now_iso(),
                    "event": "ai_reply_fallback_error",
                    "source": source,
                    "chat_id": chat_id,
                    "message_id": message.message_id,
                    "error": repr(send_exc),
                }
            )


async def handle_incoming(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    source: str,
) -> None:
    message = update.business_message if source == "business_message" else update.message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    logs: JsonlLogger = context.application.bot_data["logs"]
    message_index: Dict[str, Dict[str, Any]] = context.application.bot_data["message_index"]

    payload = message_to_dict(message, update, source)
    logs.write(payload)
    message_index[f"{source}:{message.chat_id}:{message.message_id}"] = payload

    text = extract_text(message)
    if text:
        hits = find_keyword_hits(text, settings.keywords)
        if hits:
            logs.write(
                {
                    "timestamp": utc_now_iso(),
                    "event": "keyword_hit",
                    "source": source,
                    "chat_id": message.chat_id,
                    "message_id": message.message_id,
                    "keywords": hits,
                    "text": text,
                }
            )
            await maybe_alert_admin(
                context=context,
                admin_chat_id=settings.admin_chat_id,
                text=(
                    f"Keyword hit in {source}\n"
                    f"chat_id={message.chat_id}, message_id={message.message_id}\n"
                    f"keywords={', '.join(hits)}\n\n"
                    f"text: {text[:1200]}"
                ),
            )

    if source == "message":
        if await process_prompt_input(message, context):
            return
        if await process_style_image(message, context):
            return

    await maybe_reply_with_ai(update, context, source=source)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_incoming(update, context, source="message")


async def on_business_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_incoming(update, context, source="business_message")


async def on_edited_business_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.edited_business_message
    if not message:
        return

    logs: JsonlLogger = context.application.bot_data["logs"]
    logs.write(
        {
            "timestamp": utc_now_iso(),
            "event": "edited_business_message",
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "text": message.text,
            "caption": message.caption,
        }
    )


async def on_deleted_business_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    deleted = update.deleted_business_messages
    if not deleted:
        return

    logs: JsonlLogger = context.application.bot_data["logs"]
    message_index: Dict[str, Dict[str, Any]] = context.application.bot_data["message_index"]

    deleted_payloads: List[Dict[str, Any]] = []
    for message_id in deleted.message_ids:
        cache_key = f"business_message:{deleted.chat.id}:{message_id}"
        deleted_payloads.append(
            {
                "message_id": message_id,
                "cached_message": message_index.get(cache_key),
            }
        )

    logs.write(
        {
            "timestamp": utc_now_iso(),
            "event": "deleted_business_messages",
            "business_connection_id": deleted.business_connection_id,
            "chat_id": deleted.chat.id,
            "message_ids": deleted.message_ids,
            "deleted_payloads": deleted_payloads,
        }
    )


async def on_any_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.deleted_business_messages:
        await on_deleted_business_messages(update, context)


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = Application.builder().token(settings.telegram_bot_token).build()
    state = StateStore(path=STATE_PATH)

    app.bot_data["settings"] = settings
    app.bot_data["state"] = state
    app.bot_data["logs"] = JsonlLogger(LOG_PATH)
    app.bot_data["ai"] = GeminiAssistant(
        api_key=settings.gemini_api_key,
        text_model=settings.gemini_model,
        vision_model=settings.vision_model,
        base_prompt=settings.default_prompt,
    )
    app.bot_data["message_index"] = {}
    app.bot_data["history_store"] = defaultdict(lambda: deque(maxlen=12))

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("setprompt", setprompt_command))
    app.add_handler(CommandHandler("showprompt", showprompt_command))
    app.add_handler(CommandHandler("resetprompt", resetprompt_command))
    app.add_handler(CommandHandler("learnstyle", learnstyle_command))
    app.add_handler(CommandHandler("done", done_command))
    app.add_handler(CommandHandler("showstyle", showstyle_command))
    app.add_handler(CommandHandler("clearstyle", clearstyle_command))
    app.add_handler(CommandHandler("status", status_command))

    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.Document.IMAGE, on_message))
    app.add_handler(MessageHandler(filters.UpdateType.BUSINESS_MESSAGE, on_business_message))
    app.add_handler(MessageHandler(filters.UpdateType.EDITED_BUSINESS_MESSAGE, on_edited_business_message))
    app.add_handler(TypeHandler(Update, on_any_update))

    app.run_polling(
        allowed_updates=[
            "message",
            "business_message",
            "edited_business_message",
            "deleted_business_messages",
        ]
    )


if __name__ == "__main__":
    main()
