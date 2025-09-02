# bot.py
import logging

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from main_model.sbert_predict import predict_toxic_proba

from telegram import Update, BotCommand  # NEW: BotCommand для /help-меню
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import aiohttp
import asyncio

# ───────────────────────── ЛОГИ ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("toxbot")

# ─────────────────────── НАСТРОЙКИ ───────────────────────
load_dotenv("tg_bot/api_key.env")
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Не найден API_KEY в tg_bot/api_key.env")

# ───────────────— СОСТОЯНИЕ БОТА НА КАЖДЫЙ ЧАТ ───────────────—
@dataclass
class ChatState:
    active: bool = False              # включён ли бот для этого чата
    mode: str = "all"                 # 'all' | 'only_toxic'
    threshold: float = 0.5            # порог для only_toxic

CHAT_STATES: dict[int, ChatState] = {}


def get_state(chat_id: int) -> ChatState:
    st = CHAT_STATES.get(chat_id)
    if not st:
        st = ChatState()
        CHAT_STATES[chat_id] = st
    return st


# ──────────────── УТИЛИТЫ ─────────────────
import re
import requests
from telegram import Update
from telegram.ext import ContextTypes

def get_joke_from_api() -> str:
    url = "http://rzhunemogu.ru/RandJSON.aspx?CType=1"
    try:
        resp = requests.get(url, timeout=8)
        resp.encoding = "windows-1251"

        if resp.status_code != 200:
            return "Сервер вернул ошибку 🐛"

        raw = resp.text.lstrip("\ufeff")  # убираем BOM, если есть

        # Ищем content с учётом переводов строк
        m = re.search(r'"content"\s*:\s*"(.*?)"\s*}', raw, flags=re.S)
        if not m:
            return "Не удалось распарсить анекдот 🐛"

        joke = m.group(1)
        joke = joke.replace("\r\n", "\n").replace("\r", "\n")
        joke = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', joke)

        return joke.strip() or "Пустой ответ 🐛"

    except Exception as e:
        return f"Ошибка: {e}"


# ────────────────────── ХЭНДЛЕРЫ КОМАНД ──────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    st.active = True
    st.mode = "all"
    st.threshold = 0.5
    await update.message.reply_text(
        "✅ Бот включён.\n"
        "Режим: отвечаю на ВСЕ сообщения и показываю вероятность токсичности.\n"
        "Команды: /help"
    )

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    st.active = False
    await update.message.reply_text("🛑 Бот выключен для этого чата. Команды всё ещё доступны.")

async def cmd_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    st.active = True
    st.mode = "all"
    await update.message.reply_text("🔁 Режим включён: отвечаю на все сообщения и показываю вероятность.")

async def cmd_only_toxic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    st.active = True
    st.mode = "only_toxic"

    thr = st.threshold
    if context.args:
        try:
            thr = float(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "⚠️ Порог должен быть числом, например: /only_toxic 0.6\n"
                f"Оставляю прежний порог: {st.threshold:.2f}"
            )
        else:
            thr = min(max(thr, 0.0), 1.0)
            st.threshold = thr

    await update.message.reply_text(
        f"🎯 Режим включён: отвечаю только при токсичности ≥ порога.\n"
        f"Текущий порог: {st.threshold:.2f}."
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    await update.message.reply_text(
        "ℹ️ Статус:\n"
        f"• Активен: {'да' if st.active else 'нет'}\n"
        f"• Режим: {st.mode}\n"
        f"• Порог: {st.threshold:.2f}"
    )

# NEW: /help
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    await update.message.reply_text(
        "🆘 Help\n\n"
        "Команды:\n"
        "• /start — включить бота (режим: все сообщения)\n"
        "• /stop — выключить бота\n"
        "• /all_massages — отвечать на все сообщения и выводить вероятность токсичности\n"
        "• /only_toxic <порог> — отвечать только если токсичность ≥ порога (0..1, по умолчанию 0.5)\n"
        "• /status — показать текущие настройки\n"
        "• /joke — короткий анекдот (ru)\n\n"
        f"Текущий режим: {st.mode}, порог: {st.threshold:.2f}, активен: {'да' if st.active else 'нет'}"
    )

# NEW: /joke — тянем анекдот из Shortiki
async def cmd_joke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    joke = get_joke_from_api()
    await update.message.reply_text(f"{joke}")


# ───────────────────── ОБРАБОТКА ТЕКСТА ─────────────────────
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    if not st.active:
        return

    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    try:
        proba = float(predict_toxic_proba([user_text]))
    except Exception as e:
        log.exception("predict_toxic_proba failed")
        await update.message.reply_text("❌ Ошибка модели (см. логи сервера).")
        return

    percent = f"{proba * 100:.1f}%"

    if st.mode == "all":
        await update.message.reply_text(f"🧪 Вероятность токсичности: {percent}")
    else:
        if proba >= st.threshold:
            await update.message.reply_text(
                f"⚠️ Сообщение похоже на токсичное.\n"
                f"🧪 Вероятность токсичности: {percent} (порог {st.threshold:.2f})"
            )
        else:
            pass  # молчим


# ──────────────────────── ВХОДНАЯ ТОЧКА ────────────────────────
async def _register_bot_commands(app: Application) -> None:
    """
    (необязательно) Зарегистрировать список команд в меню Telegram.
    Это даёт пользователям удобный список /команд.
    """
    commands = [
        BotCommand("start", "Включить бота (режим: все сообщения)"),
        BotCommand("stop", "Выключить бота"),
        BotCommand("all_massages", "Отвечать на все сообщения и показывать вероятность"),
        BotCommand("only_toxic", "Отвечать только при токсичности ≥ порога (0..1)"),
        BotCommand("status", "Показать текущие настройки"),
        BotCommand("joke", "Короткий анекдот (ru)"),
        BotCommand("help", "Справка по командам"),
    ]
    await app.bot.set_my_commands(commands)

def main():
    app = Application.builder().token(API_KEY).build()

    # Команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("all_massages", cmd_all_messages))
    app.add_handler(CommandHandler("all_messages", cmd_all_messages))  # синоним
    app.add_handler(CommandHandler("only_toxic", cmd_only_toxic))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))      # NEW
    app.add_handler(CommandHandler("joke", cmd_joke))      # NEW

    # Текст без команд
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    # Зарегистрируем /команды в меню (необязательно)
    app.post_init = _register_bot_commands  # NEW: вызовется при старте

    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
