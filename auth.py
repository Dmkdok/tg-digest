#!/usr/bin/env python3
"""Одноразовая авторизация Telethon-сессии. Запускать интерактивно:

    python auth.py

После первого запуска создаётся файл session.session — повторная
авторизация больше не требуется (пока сессия не отозвана).
"""
import asyncio
import os

from telethon import TelegramClient

API_ID = int(os.environ.get("TG_API_ID", "2040"))
API_HASH = os.environ.get("TG_API_HASH", "b18441a1ff607e10a989891a5462e627")


async def main():
    client = TelegramClient("session", API_ID, API_HASH)
    await client.start()
    me = await client.get_me()
    print(f"\n✅ Авторизован как: {me.first_name} (@{me.username})")
    print("Сессия сохранена в session.session — повторная авторизация не нужна.")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
