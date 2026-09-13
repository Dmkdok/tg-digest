#!/usr/bin/env python3
"""
Telegram channel digest collector + summarizer.

Reads channels from channels.txt, fetches messages from the last N hours,
summarizes them via an LLM and sends the result to Telegram.

Design notes
------------
- Map-reduce summarization: large chats (hundreds of messages) are chunked,
  each chunk is summarized, then the chunk summaries are combined into one
  final analysis. Small chats are summarized in a single ("stuff") call.
- LLM backend: a chain of providers is tried in order (see `llm_chat`) —
  free cloud models first, a paid API as an optional fallback, and a local
  Ollama instance as the last resort. The first two "cloud" calls run
  through the Hermes CLI, keeping API keys and provider wiring in one place.
- Telegram delivery: Telegram caps messages at 4096 characters, so long
  digests are split into several messages on smart boundaries (channel
  sections, then paragraphs) instead of being truncated.
"""

import asyncio
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError

# ── Telegram API credentials ─────────────────────────────────────────────────
# 2040 / b18441a1ff607e10a989891a5462e627 is the widely-reused "Telegram
# Desktop" api_id/api_hash pair (not a secret tied to any account). For your
# own bot-free/UI-free client it's fine to reuse; get your own pair at
# https://my.telegram.org if you prefer, and override via env vars below.
API_ID = int(os.environ.get("TG_API_ID", "2040"))
API_HASH = os.environ.get("TG_API_HASH", "b18441a1ff607e10a989891a5462e627")

# ── LLM backend ───────────────────────────────────────────────────────────────
# Primary/fallback models run through the Hermes CLI against Nous Portal's
# free tier. Override via env vars if you use different models/providers.
HERMES_BIN = os.environ.get("HERMES_BIN", "hermes")
NOUS_PROVIDER = os.environ.get("NOUS_PROVIDER", "nous")
NOUS_MODELS = os.environ.get(
    "NOUS_MODELS",
    "inclusionai/ling-3.0-flash-fin:free,inclusionai/ling-3.0-flash-sante:free",
).split(",")
HERMES_TIMEOUT = int(os.environ.get("HERMES_TIMEOUT", "90"))

# Optional fallback: DeepSeek (OpenAI-compatible). Activates automatically
# once DEEPSEEK_API_KEY is set.
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Last-resort fallback: local Ollama (used only if every cloud option fails).
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "")  # e.g. http://host:port/api/chat
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_KEY = os.environ.get("OLLAMA_KEY", "ollama")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

# ── Digest config ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SESSION_FILE = Path(os.environ.get("TG_SESSION_FILE", BASE_DIR / "session"))
CHANNELS_FILE = Path(os.environ.get("TG_CHANNELS_FILE", BASE_DIR / "channels.txt"))
FILLER_FILE = Path(os.environ.get("TG_FILLER_FILE", BASE_DIR / "filler_words.txt"))
TMP_QUERY_FILE = BASE_DIR / ".tmp_query.txt"

HOURS_BACK = int(os.environ.get("DIGEST_HOURS_BACK", "24"))
MAX_MSG_LEN = 300       # truncate individual messages before sending to LLM
CHUNK_SIZE = 80         # baseline messages per map-reduce chunk (quiet channels)
MAX_CHUNKS_PER_CHANNEL = 8  # cap LLM calls per channel; noisy channels get
                            # larger (coarser) chunks instead of more calls
TOP_REACTED_N = 10      # how many top-reacted messages to surface
TOP_REACTED_MIN = 3     # minimum reactions to be considered "top"

# Telethon's FloodWaitError (Telegram rate limit) tells us exactly how long to
# wait via `e.seconds`. Retry that specific error with backoff instead of
# letting the whole channel fetch fail — other errors (bad channel, network)
# still fail immediately without retry.
FLOOD_WAIT_MAX_RETRIES = 3
FLOOD_WAIT_MAX_SLEEP = 300  # cap a single wait so one channel can't stall the
                            # whole run for e.g. a 1-hour flood wait

# Telegram's hard cap is 4096 chars/message; some delivery paths don't
# auto-split long text and simply fail on oversized messages, so we split
# on our side first, with margin below the hard limit.
TELEGRAM_MAX_LEN = 3900


# ── Filtering ─────────────────────────────────────────────────────────────────

def load_filler_patterns() -> set[str]:
    """Слова/фразы без смысловой нагрузки — конфигурируются в filler_words.txt
    (по одному на строку, пустые строки/# — игнорируются), без правки кода."""
    if not FILLER_FILE.exists():
        return set()
    with open(FILLER_FILE, encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


FILLER_PATTERNS = load_filler_patterns()


def is_filler(text: str) -> bool:
    """True если сообщение — короткая реплика без смысловой нагрузки
    (спасибо/ок/эмодзи-реакция и т.п.), которую не стоит слать в LLM."""
    t = text.strip().lower().strip(".!?,;: \n")
    if not t:
        return True
    if t in FILLER_PATTERNS:
        return True
    words = t.split()
    return len(words) <= 4 and all(w.strip("!?.,;:()") in FILLER_PATTERNS for w in words)


def load_channels() -> list[str | int]:
    """Каналы задаются @username (публичные) или числовым ID вида -100...
    (приватные чаты/супергруппы, доступные только по t.me/c/<id>/<msg>)."""
    channels = []
    with open(CHANNELS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            channels.append(int(line) if line.lstrip("-").isdigit() else line.lstrip("@"))
    return channels


# ── Telegram fetch ────────────────────────────────────────────────────────────

def _reaction_count(msg) -> int:
    if not getattr(msg, "reactions", None) or not msg.reactions.results:
        return 0
    return sum(r.count for r in msg.reactions.results)


async def fetch_channel(client, channel: str, since: datetime) -> dict:
    title = channel
    for attempt in range(FLOOD_WAIT_MAX_RETRIES + 1):
        try:
            entity = await client.get_entity(channel)
            title = getattr(entity, "title", channel)
            messages = []
            async for msg in client.iter_messages(entity, offset_date=datetime.now(timezone.utc)):
                if msg.date < since:
                    break
                if msg.text and msg.text.strip() and not is_filler(msg.text):
                    messages.append({"text": msg.text.strip(), "reactions": _reaction_count(msg)})
            messages.reverse()
            return {"channel": channel, "title": title, "messages": messages}
        except FloodWaitError as e:
            if attempt >= FLOOD_WAIT_MAX_RETRIES:
                return {
                    "channel": channel, "title": title, "messages": [],
                    "error": f"Telegram rate limit (FloodWait): исчерпаны попытки "
                             f"после {FLOOD_WAIT_MAX_RETRIES} ретраев, требуется "
                             f"ждать ещё {e.seconds}с",
                }
            await asyncio.sleep(min(e.seconds, FLOOD_WAIT_MAX_SLEEP))
            # restart the fetch from scratch next loop — cheap, messages are
            # re-filtered by `since` anyway (no duplicates)
        except Exception as e:
            return {"channel": channel, "title": title, "messages": [], "error": str(e)}


# ── LLM backends ──────────────────────────────────────────────────────────────

def _clip(t: str, max_len: int = MAX_MSG_LEN) -> str:
    return t if len(t) <= max_len else t[:max_len] + "…"


def hermes_chat(prompt: str, model: str) -> str:
    """Run a single query through the Hermes CLI. Uses --query-file to avoid
    shell-escaping issues with large/multiline text."""
    TMP_QUERY_FILE.write_text(prompt, encoding="utf-8")
    result = subprocess.run(
        [
            HERMES_BIN, "chat",
            "-m", model,
            "--provider", NOUS_PROVIDER,
            "--query-file", str(TMP_QUERY_FILE),
            "--oneshot", "-Q",
            "-t", "",             # no tools needed, keeps it fast/cheap
            "--source", "tool",   # tag as automated, keeps `hermes sessions
                                  # list` clean for daily/frequent runs
        ],
        capture_output=True, text=True, timeout=HERMES_TIMEOUT,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"hermes chat ({model}) failed: rc={result.returncode} "
                            f"stderr={result.stderr[:300]} stdout={result.stdout[:300]}")
    lines = (l for l in result.stdout.strip().splitlines() if not l.strip().startswith("session_id:"))
    text = "\n".join(lines).strip()
    if not text:
        raise RuntimeError(f"hermes chat ({model}) returned empty content")
    return text


def deepseek_chat(prompt: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set — skipping DeepSeek fallback")
    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        DEEPSEEK_API_URL, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def ollama_chat(prompt: str) -> str:
    if not OLLAMA_CHAT_URL:
        raise RuntimeError("OLLAMA_CHAT_URL not set — skipping local Ollama fallback")
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.3},
        "keep_alive": "10m",
    }).encode()
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {OLLAMA_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"].strip()


def llm_chat(prompt: str) -> str:
    """Tries each backend in order, cloud-first, and raises with the combined
    error trail only if every one of them failed."""
    errors = []
    for model in NOUS_MODELS:
        try:
            return hermes_chat(prompt, model)
        except Exception as e:
            errors.append(f"{model}: {e}")
    for backend, fn in (("deepseek", deepseek_chat), ("local ollama", ollama_chat)):
        try:
            return fn(prompt)
        except Exception as e:
            errors.append(f"{backend}: {e}")
    raise RuntimeError("All LLM backends failed:\n" + "\n".join(errors))


# ── Summarization ─────────────────────────────────────────────────────────────

REPORT_STRUCTURE = (
    "### Темы и события\n"
    "3-6 предложений о главных темах и событиях дня{extra}.\n\n"
    "### Выводы и тезисы\n"
    "Если участники в ходе обсуждения пришли к общему мнению, консенсусу или практическому "
    "выводу (совет, решение, факт, рекомендация) — перечисли списком, конкретно и по делу. "
    "Если ничего подобного не было — напиши «выводов не было», не выдумывай.\n\n"
    "### Полезное по реакциям\n"
    "Из списка сообщений с наибольшим числом реакций (ниже) выбери только те, что реально "
    "полезны, интересны или познавательны, и кратко опиши суть с числом реакций. "
    "Бытовые шутки, сленг и склоки пропусти. Если ничего полезного нет — напиши "
    "«ничего полезного не было», не выдумывай.\n\n"
    "Пиши компактно, без вступлений и воды.\n\n"
)


def _top_reacted_block(top_reacted: list[dict]) -> str:
    if not top_reacted:
        return f"нет сообщений с {TOP_REACTED_MIN}+ реакциями"
    return "\n---\n".join(f"[{m['reactions']} реакций] {_clip(m['text'])}" for m in top_reacted)


def summarize_chunk(title: str, chunk: list[dict], part_no: int, total_parts: int, clip_len: int) -> str:
    text = "\n---\n".join(_clip(m["text"], clip_len) for m in chunk)
    prompt = (
        f"Это часть {part_no} из {total_parts} переписки Telegram-чата «{title}» "
        f"за последние {HOURS_BACK} ч (сообщения идут в хронологическом порядке).\n"
        f"Кратко на русском языке перечисли:\n"
        f"- о каких темах/событиях шла речь в этой части;\n"
        f"- если участники пришли к какому-то мнению, выводу, решению или совету — укажи это отдельно.\n"
        f"Пиши тезисно, без вступлений и заключений, 5-10 пунктов максимум.\n\n"
        f"{text}"
    )
    return llm_chat(prompt)


def synthesize_final(title: str, chunk_summaries: list[str], top_reacted: list[dict]) -> str:
    joined = "\n\n".join(f"[Часть {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries))
    structure = REPORT_STRUCTURE.format(extra=" (объедини повторяющиеся темы из разных частей)")
    prompt = (
        f"Ниже — тезисные саммари последовательных частей обсуждения в Telegram-чате «{title}» "
        f"за последние {HOURS_BACK} ч, а также список сообщений с наибольшим числом реакций.\n\n"
        f"На основе ВСЕГО материала сделай итоговый разбор на русском языке по структуре:\n\n"
        f"{structure}"
        f"=== Саммари частей ===\n{joined}\n\n"
        f"=== Сообщения с наибольшим числом реакций ===\n{_top_reacted_block(top_reacted)}"
    )
    return llm_chat(prompt)


def summarize_stuff(title: str, messages: list[dict], top_reacted: list[dict]) -> str:
    """Один вызов LLM на весь чат целиком (без map-reduce) — используется,
    когда сообщения помещаются в один чанк: быстрее и без потерь на границах."""
    text = "\n---\n".join(_clip(m["text"]) for m in messages)
    structure = REPORT_STRUCTURE.format(extra="")
    prompt = (
        f"Ниже — переписка Telegram-чата «{title}» за последние {HOURS_BACK} ч "
        f"(сообщения идут в хронологическом порядке), а также список сообщений "
        f"с наибольшим числом реакций.\n\n"
        f"На основе ВСЕГО материала сделай разбор на русском языке по структуре:\n\n"
        f"{structure}"
        f"=== Переписка ===\n{text}\n\n"
        f"=== Сообщения с наибольшим числом реакций ===\n{_top_reacted_block(top_reacted)}"
    )
    return llm_chat(prompt)


def summarize(title: str, messages: list[dict]) -> str:
    if not messages:
        return f"Сообщений за последние {HOURS_BACK} ч не было."

    top_reacted = sorted(
        (m for m in messages if m["reactions"] >= TOP_REACTED_MIN), key=lambda m: -m["reactions"]
    )[:TOP_REACTED_N]

    # "stuff": чат укладывается в один базовый чанк — суммируем целиком одним
    # вызовом (быстрее и без потерь map-reduce на границах чанков).
    if len(messages) <= CHUNK_SIZE:
        return summarize_stuff(title, messages, top_reacted)

    # Адаптивный размер чанка: тихим каналам — CHUNK_SIZE (детальные саммари),
    # шумным — крупнее чанк, чтобы не раздувать число вызовов LLM выше
    # MAX_CHUNKS_PER_CHANNEL (грубее детализация, зато быстро).
    chunk_size = max(CHUNK_SIZE, -(-len(messages) // MAX_CHUNKS_PER_CHANNEL))
    chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
    # В шумных чатах отдельные сообщения тоже режем короче, чтобы промпт
    # крупного чанка не раздувался пропорционально числу сообщений.
    clip_len = MAX_MSG_LEN if chunk_size <= CHUNK_SIZE else max(80, MAX_MSG_LEN * CHUNK_SIZE // chunk_size)
    chunk_summaries = [
        summarize_chunk(title, chunk, i + 1, len(chunks), clip_len) for i, chunk in enumerate(chunks)
    ]
    return synthesize_final(title, chunk_summaries, top_reacted)


# ── Digest assembly & delivery ────────────────────────────────────────────────

async def build_digest() -> list[str]:
    """Возвращает список секций (баннер + одна секция на канал), а не единую
    строку — Telegram ограничивает сообщение 4096 символами, и сплит нужен по
    смысловым границам (см. split_into_messages), а не вслепую."""
    channels = load_channels()
    since = datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)

    async with TelegramClient(str(SESSION_FILE), API_ID, API_HASH) as client:
        results = [await fetch_channel(client, ch, since) for ch in channels]

    date_str = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    sections = [f"📋 **Telegram-дайджест за {date_str}**"]

    for r in results:
        ch_label = f"@{r['channel']}" if isinstance(r["channel"], str) else f"id{r['channel']}"
        part = [f"## {r['title']} ({ch_label})"]
        if r.get("error"):
            part.append(f"⚠️ Ошибка: {r['error']}")
        else:
            part.append(f"*{len(r['messages'])} сообщений за {HOURS_BACK} ч*\n")
            try:
                part.append(summarize(r["title"], r["messages"]))
            except Exception as e:
                part.append(f"⚠️ Не удалось построить саммари: {e}")
        sections.append("\n".join(part))

    TMP_QUERY_FILE.unlink(missing_ok=True)
    return sections


def split_into_messages(sections: list[str], max_len: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Пакует секции (по каналу) в сообщения по max_len символов, стараясь не
    рвать секцию посередине; если секция сама больше max_len — режет по абзацам
    (и, в крайнем случае, жёстко по длине)."""
    messages: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            messages.append(current.rstrip())
            current = ""

    def add(piece: str):
        nonlocal current
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_len:
            current = candidate
            return True
        return False

    for section in sections:
        if add(section):
            continue
        flush()
        if add(section):
            continue
        # секция сама больше лимита — режем по абзацам, затем жёстко по длине
        for para in section.split("\n\n"):
            if add(para):
                continue
            flush()
            if add(para):
                continue
            for i in range(0, len(para), max_len):
                messages.append(para[i:i + max_len])
    flush()
    return messages


def send_telegram(text: str) -> None:
    result = subprocess.run(
        [HERMES_BIN, "send", "--to", "telegram", "--quiet", text],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"hermes send failed: rc={result.returncode} stderr={result.stderr[:300]}")


async def main():
    # Top-level safety net: this runs unattended via cron, so any exception
    # must still produce *something* to send — silent stdout means the cron
    # job silently sends nothing at all.
    try:
        messages = split_into_messages(await build_digest())
    except Exception as e:
        date_str = datetime.now(timezone.utc).strftime("%d.%m.%Y")
        messages = [
            f"⚠️ **Telegram-дайджест за {date_str} не собран**\n\n"
            f"Скрипт fetch_digest.py упал с ошибкой:\n```\n{e}\n```"
        ]

    failures = []
    for i, msg in enumerate(messages, 1):
        try:
            send_telegram(msg)
        except Exception as e:
            failures.append(f"часть {i}/{len(messages)}: {e}")

    print(f"Дайджест отправлен: {len(messages)} сообщени{'е' if len(messages) == 1 else 'й'}.")
    if failures:
        print("⚠️ Ошибки отправки:\n" + "\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
