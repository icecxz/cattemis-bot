import asyncio
import json
import os
import re
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import yt_dlp
from dotenv import load_dotenv
from openai import AsyncOpenAI
from yt_dlp.utils import DownloadError

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import (
    FSInputFile,
    InputMediaPhoto,
    InputMediaVideo,
    Message,
    ReplyParameters,
)
from aiogram.utils.chat_action import ChatActionSender

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
APIFY_INSTAGRAM_ACTOR = os.getenv("APIFY_INSTAGRAM_ACTOR", "elis~instagram-downloader-api")

LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b")
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "Ты милый телеграм-бот. Отвечай кратко, дружелюбно и по делу. "
    "Не выдумывай факты. Если не уверен — честно скажи об этом.",
)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не найден в .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

llm_client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
) if LLM_ENABLED else None

TIKWM_API = "https://www.tikwm.com/api/"

TIKTOK_DOMAINS = {
    "tiktok.com",
    "www.tiktok.com",
    "m.tiktok.com",
    "vm.tiktok.com",
    "vt.tiktok.com",
}

INSTAGRAM_DOMAINS = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
}

TWITTER_DOMAINS = {
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "mobile.twitter.com",
}

YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
}

VIMEO_DOMAINS = {
    "vimeo.com",
    "www.vimeo.com",
}

ALLOWED_MEDIA_HOSTS = (
    TIKTOK_DOMAINS
    | INSTAGRAM_DOMAINS
    | TWITTER_DOMAINS
    | YOUTUBE_DOMAINS
    | VIMEO_DOMAINS
)

GROUP_CHAT_TYPES = {"group", "supergroup"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".ogg", ".wav", ".flac"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

MEDIA_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}

RETRY_ATTEMPTS = 2
RETRY_DELAY = 1.2

ADMIN_CACHE_TTL = 60
MAX_HISTORY_MESSAGES = 8

_admin_cache: dict[int, tuple[float, set[int]]] = {}
_chat_locks: dict[int, asyncio.Lock] = {}
_chat_histories: dict[int, list[dict[str, str]]] = {}

_api_lock = asyncio.Lock()
_last_api_call: float = 0.0

BOT_USERNAME_CACHE: str | None = None
BOT_ID_CACHE: int | None = None

PRAISE_REPLIES = [
    "ананас",
]

PRAISE_KEYWORDS = [
    "огурец",
]

ARTISTS_CONFIG_PATH = Path(__file__).parent / "artists.json"


@dataclass
class ArtistLink:
    artist_id: str
    label: str
    url: str


_artists_cache: list[ArtistLink] = []

_stats = {
    "started_at": time.time(),
    "messages_total": 0,
    "commands_used": 0,
    "llm_calls": 0,
    "llm_errors": 0,
    "media_total": 0,
    "media_errors": 0,
    "tiktok_downloads": 0,
    "instagram_downloads": 0,
    "twitter_downloads": 0,
    "direct_image_downloads": 0,
    "ytdlp_downloads": 0,
    "unique_chats": set(),
}


def get_chat_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _chat_locks:
        _chat_locks[chat_id] = asyncio.Lock()
    return _chat_locks[chat_id]


def get_chat_history(chat_id: int) -> list[dict[str, str]]:
    if chat_id not in _chat_histories:
        _chat_histories[chat_id] = []
    return _chat_histories[chat_id]


def append_chat_history(chat_id: int, role: str, content: str) -> None:
    history = get_chat_history(chat_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_MESSAGES:
        _chat_histories[chat_id] = history[-MAX_HISTORY_MESSAGES:]


def clear_chat_history(chat_id: int) -> None:
    _chat_histories.pop(chat_id, None)

def stat_inc(key: str, value: int = 1) -> None:
    _stats[key] = _stats.get(key, 0) + value

def stat_track_chat(chat_id: int) -> None:
    _stats["unique_chats"].add(chat_id)

def format_uptime(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}ч {m}м {s}с"

async def tg_call(func, *args, retries: int = 3, **kwargs):
    for attempt in range(retries + 1):
        try:
            return await func(*args, **kwargs)
        except TelegramRetryAfter as e:
            if attempt >= retries:
                raise
            await asyncio.sleep(float(e.retry_after) + 0.5)


async def safe_status_edit(status: Message, text: str) -> None:
    try:
        await tg_call(status.edit_text, text)
    except Exception:
        pass


async def safe_delete_message(message: Message | None):
    if not message:
        return
    try:
        await tg_call(message.delete)
    except Exception:
        pass


async def rate_limit_free_api() -> None:
    global _last_api_call
    async with _api_lock:
        now = time.monotonic()
        diff = now - _last_api_call
        if diff < 1.1:
            await asyncio.sleep(1.1 - diff)
        _last_api_call = time.monotonic()


async def get_bot_username() -> str:
    global BOT_USERNAME_CACHE
    if BOT_USERNAME_CACHE is None:
        me = await bot.get_me()
        BOT_USERNAME_CACHE = (me.username or "").lower()
    return BOT_USERNAME_CACHE


async def get_bot_id() -> int:
    global BOT_ID_CACHE
    if BOT_ID_CACHE is None:
        me = await bot.get_me()
        BOT_ID_CACHE = me.id
    return BOT_ID_CACHE


def normalize_possible_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def host_matches(host: str, allowed_hosts: set[str]) -> bool:
    host = (host or "").lower()
    return any(host == item or host.endswith("." + item) for item in allowed_hosts)


def is_tiktok(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host_matches(host, TIKTOK_DOMAINS)


def is_instagram(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host_matches(host, INSTAGRAM_DOMAINS)


def is_twitter(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host_matches(host, TWITTER_DOMAINS)


def is_direct_image(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return False
    return any(path.endswith(ext) for ext in IMAGE_EXTS)


def is_allowed_media_link(url: str) -> bool:
    url = normalize_possible_url(url)

    try:
        parsed = urlparse(url)
    except Exception:
        return False

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    if host_matches(host, ALLOWED_MEDIA_HOSTS):
        return True

    if any(path.endswith(ext) for ext in MEDIA_EXTS):
        return True

    return False


def guess_ext_from_content_type(content_type: str | None, fallback: str = ".jpg") -> str:
    if not content_type:
        return fallback
    content_type = content_type.split(";")[0].strip().lower()
    return MEDIA_CONTENT_TYPES.get(content_type, fallback)


def extract_urls_from_message(message: Message) -> list[str]:
    text = message.text or message.caption or ""
    entities = message.entities or message.caption_entities or []
    urls = []

    for entity in entities:
        entity_type = str(entity.type)

        if entity_type == "url":
            raw = text[entity.offset: entity.offset + entity.length]
            raw = normalize_possible_url(raw)
            if raw:
                urls.append(raw)

        elif entity_type == "text_link" and entity.url:
            raw = normalize_possible_url(entity.url)
            if raw:
                urls.append(raw)

    dedup = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            dedup.append(url)

    return dedup


def is_praise_text(text: str) -> bool:
    if not text:
        return False
    normalized = " ".join(text.lower().strip().split())
    return any(keyword in normalized for keyword in PRAISE_KEYWORDS)


async def is_reply_to_this_bot(message: Message) -> bool:
    reply = message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return reply.from_user.id == await get_bot_id()


async def is_bot_mentioned(message: Message) -> bool:
    text = message.text or message.caption or ""
    entities = message.entities or message.caption_entities or []

    bot_username = await get_bot_username()
    if not bot_username:
        return False

    expected = f"@{bot_username}"

    for entity in entities:
        if str(entity.type) == "mention":
            mention_text = text[entity.offset: entity.offset + entity.length].lower()
            if mention_text == expected:
                return True

    return False


async def is_praise_for_bot(message: Message) -> bool:
    raw_text = (message.text or message.caption or "").strip()
    if not raw_text or not is_praise_text(raw_text):
        return False

    if await is_reply_to_this_bot(message):
        return True

    if await is_bot_mentioned(message):
        return True

    return False


async def get_admin_ids(chat_id: int) -> set[int]:
    now = time.monotonic()
    cached = _admin_cache.get(chat_id)

    if cached:
        ts, ids = cached
        if now - ts < ADMIN_CACHE_TTL:
            return ids

    admins = await bot.get_chat_administrators(chat_id)
    ids = {member.user.id for member in admins}
    _admin_cache[chat_id] = (now, ids)
    return ids


async def is_admin_message(message: Message) -> bool:
    if not message.from_user:
        return False
    if message.chat.type not in GROUP_CHAT_TYPES:
        return False

    admin_ids = await get_admin_ids(message.chat.id)
    return message.from_user.id in admin_ids


async def can_use_say(message: Message) -> bool:
    if message.chat.type == "private":
        return True
    return await is_admin_message(message)


async def moderate_links(message: Message) -> tuple[bool, list[str]]:
    urls = extract_urls_from_message(message)
    if not urls:
        return False, []

    if message.chat.type not in GROUP_CHAT_TYPES:
        return False, urls

    if await is_admin_message(message):
        return False, urls

    has_bad_links = any(not is_allowed_media_link(url) for url in urls)
    if has_bad_links:
        try:
            await message.delete()
        except Exception:
            pass
        return True, urls

    return False, urls


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True

    if isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientOSError,
        ),
    ):
        return True

    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in {429, 500, 502, 503, 504}

    text = str(exc).lower()
    retry_markers = [
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "server disconnected",
        "too many requests",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    ]
    return any(marker in text for marker in retry_markers)


async def with_retry(func, *args, attempts: int = RETRY_ATTEMPTS, delay: float = RETRY_DELAY, **kwargs):
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt >= attempts or not is_retryable_exception(e):
                raise
            await asyncio.sleep(delay)

    raise last_error


def human_ytdlp_error(error: Exception) -> str:
    text = str(error).lower()

    if "requested format is not available" in text:
        return "Хозяин, нужна другая ссылка. Такое качество не помещается в мой животик (⁠╯⁠︵⁠╰⁠,⁠)."
    if "unsupported url" in text or "not a valid url" in text:
        return "Хозяин, кажется, эта ссылка нерабочая (⁠˘⁠･⁠_⁠･⁠˘⁠)"
    if "private video" in text:
        return "Хозяин, это видео приватное и я не могу его посмотреть (⁠￣⁠ヘ⁠￣⁠;⁠)"
    if "sign in to confirm your age" in text or "age-restricted" in text:
        return "Хозяин, не подумай ничего плохого, но я не могу скачивать видео с ограничениями по возрасту... (⁠；⁠^⁠ω⁠^⁠）"
    if "video unavailable" in text:
        return "Ой, Хозяин, это видео уже недоступно (⁠･⁠o⁠･⁠;⁠)"
    if "http error 403" in text or "forbidden" in text:
        return "Ай, сайт не разрешил скачать видео. :3"
    if "timed out" in text:
        return "Мм, сервер отвечает слишком долго... Попробуй ещё разочек чуть позже, зайка ^^"
    return "Не получилось скачать это видео."


def human_instagram_api_error(error: Exception) -> str:
    text = str(error).lower()

    if "apify_token" in text:
        return "П-простите, хозяин... APIFY_TOKEN не задан... ૮(˶ㅠ︿ㅠ)ა"
    if "http 401" in text or "http 403" in text:
        return "П-простите, хозяин... Apify не принял токен... ૮(˶ㅠ︿ㅠ)ა"
    if "http 402" in text:
        return "Хозяин... у Apify, похоже, закончился баланс или лимит... ૮(˶ㅠ︿ㅠ)ა"
    if "не вернул результатов" in text:
        return "Хозяин... Apify ничего не нашёл по этой ссылке... простите... ૮(˶ㅠ︿ㅠ)ა"
    if "не вернул прямые ссылки" in text:
        return "Хозяин... Apify обработал ссылку, но не отдал прямые ссылки на медиа... ૮(˶ㅠ︿ㅠ)ა"
    if "не удалось скачать медиафайлы" in text:
        return "Хозяин... ссылки достать получилось, н-но сами файлы скачать не вышло... ૮(˶ㅠ︿ㅠ)ა"
    if "timed out" in text:
        return "Хозяин... Instagram через Apify отвечает слишком долго... попробуйте ещё разочек... ૮(˶ㅠ︿ㅠ)ა"

    return "П-простите, хозяин... не получилось скачать Instagram через Apify... ૮(˶ㅠ︿ㅠ)ა"


def human_twitter_error(error: Exception) -> str:
    text = str(error).lower()

    if "распарсить ссылку" in text:
        return "П-простите, хозяин... я не понял ссылку на пост... ૮(˶ㅠ︿ㅠ)ა"
    if "не вернул медиа" in text:
        return "Хозяин... в этом посте не нашлось медиа... ૮(˶ㅠ︿ㅠ)ა"
    if "404" in text:
        return "Хозяин... пост не найден или его уже удалили... ૮(˶ㅠ︿ㅠ)ა"
    if "403" in text:
        return "П-простите, хозяин... тивттер не отдал данные по этому посту... ( . ‸ .)"
    if "timed out" in text:
        return "Хозяин... твиттер отвечает слишком долго... попробуйте ещё разочек... ( . ‸ .)"

    return "П-простите, хозяин... не получилось скачать медиа из твиттера... ૮(˶ㅠ︿ㅠ)ა"


async def download_tiktok(url: str) -> dict:
    await rate_limit_free_api()

    timeout = aiohttp.ClientTimeout(total=40)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        )
    }

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.post(TIKWM_API, data={"url": url, "hd": 1}) as resp:
            resp.raise_for_status()
            payload = await resp.json(content_type=None)

    if payload.get("code") != 0 or not payload.get("data"):
        raise RuntimeError(payload.get("msg") or "TikWM не вернул данные")

    return payload["data"]


def extract_media_urls(obj) -> list[str]:
    found = []
    skip_keys = {
        "thumbnail",
        "thumb",
        "avatar",
        "profile",
        "icon",
        "logo",
        "permalink",
        "shortcode",
        "posturl",
        "pageurl",
    }
    good_keys = {
        "video",
        "image",
        "photo",
        "display",
        "download",
        "src",
        "media",
        "url",
        "play",
    }

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                key = str(k).lower()

                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    if any(bad in key for bad in skip_keys):
                        pass
                    elif any(ok in key for ok in good_keys):
                        found.append(v)

                walk(v)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(obj)

    dedup = []
    seen = set()
    for url in found:
        if url not in seen:
            seen.add(url)
            dedup.append(url)

    return dedup


async def download_instagram_apify(url: str) -> dict:
    if not APIFY_TOKEN:
        raise RuntimeError("APIFY_TOKEN не задан")

    endpoint = f"https://api.apify.com/v2/acts/{APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items"
    payload = {"url": [url]}
    headers = {
        "Authorization": f"Bearer {APIFY_TOKEN}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=90)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            raw_text = await resp.text()

            if resp.status >= 400:
                raise RuntimeError(f"Apify HTTP {resp.status}: {raw_text[:300]}")

            try:
                data = await resp.json(content_type=None)
            except Exception:
                raise RuntimeError("Apify вернул не JSON")

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or not data:
        raise RuntimeError("Apify не вернул результатов")

    caption = None
    media_urls = []

    for item in data:
        if isinstance(item, dict):
            if not caption:
                for key in ("caption", "title", "text"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        caption = value.strip()
                        break

            media_urls.extend(extract_media_urls(item))

    dedup_urls = []
    seen = set()
    for media_url in media_urls:
        if media_url not in seen:
            seen.add(media_url)
            dedup_urls.append(media_url)

    if not dedup_urls:
        raise RuntimeError("Apify не вернул прямые ссылки на медиа")

    temp_dir = tempfile.mkdtemp(prefix="apify_ig_")
    files = []

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, media_url in enumerate(dedup_urls[:10], start=1):
                async with session.get(media_url) as resp:
                    resp.raise_for_status()

                    content_type = (resp.headers.get("Content-Type") or "").split(";")[0].lower()
                    if not (
                        content_type.startswith("image/")
                        or content_type.startswith("video/")
                        or content_type.startswith("audio/")
                    ):
                        continue

                    parsed = urlparse(media_url)
                    path_ext = Path(parsed.path).suffix.lower()
                    ext = path_ext if path_ext in MEDIA_EXTS else guess_ext_from_content_type(content_type)

                    file_path = Path(temp_dir) / f"ig_{i}{ext}"
                    file_path.write_bytes(await resp.read())
                    files.append(str(file_path))

        if not files:
            raise RuntimeError("Не удалось скачать медиафайлы из ссылок Apify")

        return {
            "temp_dir": temp_dir,
            "files": files,
            "caption": caption,
        }
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def parse_twitter_url(url: str) -> tuple[str, str] | None:
    try:
        parts = [p for p in urlparse(url).path.split("/") if p]
    except Exception:
        return None

    if len(parts) >= 3 and parts[1] == "status":
        return parts[0], parts[2]

    return None


def extract_twitter_media_urls(payload) -> list[str]:
    found = []

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                key = str(k).lower()

                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    if any(x in key for x in ["url", "media", "image", "photo", "video", "playback", "source", "src"]):
                        found.append(v)

                walk(v)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    tweet = payload.get("tweet", payload)
    media = tweet.get("media", {}) if isinstance(tweet, dict) else {}
    walk(media)

    result = []
    seen = set()
    for url in found:
        if url not in seen:
            seen.add(url)
            result.append(url)

    return result


async def download_twitter_fx(url: str) -> dict:
    parsed = parse_twitter_url(url)
    if not parsed:
        raise RuntimeError("Не удалось распарсить ссылку Twitter/X")

    username, tweet_id = parsed
    endpoint = f"https://api.fxtwitter.com/{username}/status/{tweet_id}"
    timeout = aiohttp.ClientTimeout(total=40)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(endpoint) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

    tweet = data.get("tweet", data)
    caption = None
    if isinstance(tweet, dict):
        text = tweet.get("text")
        if isinstance(text, str) and text.strip():
            caption = text.strip()

    media_urls = extract_twitter_media_urls(data)
    if not media_urls:
        raise RuntimeError("FxTwitter не вернул медиа")

    temp_dir = tempfile.mkdtemp(prefix="twitter_fx_")
    files = []

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, media_url in enumerate(media_urls[:10], start=1):
                async with session.get(media_url) as resp:
                    resp.raise_for_status()

                    content_type = (resp.headers.get("Content-Type") or "").split(";")[0].lower()
                    if not (
                        content_type.startswith("image/")
                        or content_type.startswith("video/")
                        or content_type.startswith("audio/")
                    ):
                        continue

                    parsed_media = urlparse(media_url)
                    path_ext = Path(parsed_media.path).suffix.lower()
                    ext = path_ext if path_ext in MEDIA_EXTS else guess_ext_from_content_type(content_type)

                    file_path = Path(temp_dir) / f"tw_{i}{ext}"
                    file_path.write_bytes(await resp.read())
                    files.append(str(file_path))

        if not files:
            raise RuntimeError("Не удалось скачать файлы из медиа-ссылок FxTwitter")

        return {
            "temp_dir": temp_dir,
            "files": files,
            "caption": caption,
        }
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


async def download_direct_image(url: str) -> dict:
    temp_dir = tempfile.mkdtemp(prefix="img_bot_")
    parsed = urlparse(url)
    path_ext = Path(parsed.path).suffix.lower()
    timeout = aiohttp.ClientTimeout(total=40)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type")
                ext = path_ext if path_ext in IMAGE_EXTS else guess_ext_from_content_type(content_type, ".jpg")
                file_path = Path(temp_dir) / f"image{ext}"
                file_path.write_bytes(await resp.read())

        return {"temp_dir": temp_dir, "files": [str(file_path)], "caption": None}
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


async def download_ytdlp(url: str) -> dict:
    temp_dir = tempfile.mkdtemp(prefix="media_bot_")
    outtmpl = str(Path(temp_dir) / "%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bv*[height<=1080]+ba/b[height<=1080]",
        "noplaylist": True,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    def _download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        files = []
        for p in Path(temp_dir).iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            if p.suffix.lower() in {".part", ".ytdl", ".temp"}:
                continue
            files.append(p)

        if not files:
            raise FileNotFoundError("Файл после скачивания не найден")

        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return {
            "temp_dir": temp_dir,
            "files": [str(files[0])],
            "caption": None,
        }

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _download)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


async def send_local_media(message: Message, files: list[str], caption: str | None = None):
    files = files[:10]
    if not files:
        raise RuntimeError("Нет файлов для отправки")

    caption = (caption or "").strip()[:1024] or None

    if len(files) == 1:
        path = files[0]
        ext = Path(path).suffix.lower()

        if ext in IMAGE_EXTS:
            await tg_call(message.answer_photo, FSInputFile(path), caption=caption)
            return

        if ext in VIDEO_EXTS:
            await tg_call(
                message.answer_video,
                FSInputFile(path),
                caption=caption,
                supports_streaming=True,
            )
            return

        if ext in AUDIO_EXTS:
            await tg_call(message.answer_audio, FSInputFile(path), caption=caption)
            return

        await tg_call(message.answer_document, FSInputFile(path), caption=caption)
        return

    album = []
    leftovers = []

    for path in files:
        ext = Path(path).suffix.lower()
        item_caption = caption if len(album) == 0 and caption else None

        if ext in IMAGE_EXTS:
            album.append(InputMediaPhoto(media=FSInputFile(path), caption=item_caption))
        elif ext in VIDEO_EXTS:
            album.append(InputMediaVideo(media=FSInputFile(path), caption=item_caption))
        else:
            leftovers.append(path)

    if album:
        await tg_call(message.answer_media_group, media=album)

    for i, path in enumerate(leftovers):
        ext = Path(path).suffix.lower()
        item_caption = caption if not album and i == 0 else None

        if ext in AUDIO_EXTS:
            await tg_call(message.answer_audio, FSInputFile(path), caption=item_caption)
        else:
            await tg_call(message.answer_document, FSInputFile(path), caption=item_caption)


def load_artists_config() -> None:
    global _artists_cache

    if not ARTISTS_CONFIG_PATH.exists():
        print(f"☆ Artists - {ARTISTS_CONFIG_PATH} not found, /art won't work")
        _artists_cache = []
        return

    with ARTISTS_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    artists = data.get("artists") or []
    links: list[ArtistLink] = []

    for artist in artists:
        if not artist.get("enabled", True):
            continue

        artist_id = str(artist.get("id") or "").strip()
        label = str(artist.get("label") or artist_id or "artist").strip()
        urls = artist.get("urls") or []

        for raw_url in urls:
            url = normalize_possible_url(str(raw_url))
            if not url:
                continue
            links.append(ArtistLink(artist_id=artist_id, label=label, url=url))

    _artists_cache = links
    print(f"☆ Artists loaded, {len(_artists_cache)} links")


def random_artist_link(artist_id: str | None = None) -> ArtistLink | None:
    if not _artists_cache:
        return None

    if artist_id:
        candidates = [l for l in _artists_cache if l.artist_id.lower() == artist_id.lower()]
        if not candidates:
            return None
        return random.choice(candidates)

    return random.choice(_artists_cache)


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


def strip_unicode_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text)


def cleanup_llm_text(text: str) -> str:
    text = strip_unicode_emoji(text)
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ?([,.;:!?]){2,}", r"\1", text)
    return text.strip()

def fix_truncated_kaomoji(text: str) -> str:
    if not text:
        return text

    if re.search(r">\/{2,}$", text):
        text += "<"

    if text.endswith(">///"):
        text += "/<"

    return text


async def ask_llm(chat_id: int, user_text: str, user_name: str | None = None) -> str:
    if not LLM_ENABLED or llm_client is None:
        return "LLM отключён."

    display_name = (user_name or "user").strip() or "user"
    history = get_chat_history(chat_id)
    user_content = f"{display_name}: {user_text}"

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]

    response = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=100,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    choice = response.choices[0]
    print(f"[llm] finish_reason={choice.finish_reason!r}")

    text = choice.message.content or ""
    text = cleanup_llm_text(text)
    text = fix_truncated_kaomoji(text)

    append_chat_history(chat_id, "user", user_content)
    append_chat_history(chat_id, "assistant", text or "...")

    return text or "..."

async def process_media_url(message: Message, url: str, initial_status_text: str = "Скачиваю..."):
    status = await tg_call(message.answer, initial_status_text)
    temp_dirs: list[str] = []

    try:
        if is_tiktok(url):
            data = await with_retry(download_tiktok, url)
            title = (data.get("title") or "").strip()
            stat_inc("media_total")
            stat_inc("tiktok_downloads")

            async with get_chat_lock(message.chat.id):
                images = data.get("images") or []
                if images:
                    media = [
                        InputMediaPhoto(
                            media=img,
                            caption=title[:1024] if i == 0 and title else None,
                        )
                        for i, img in enumerate(images[:10])
                    ]
                    await tg_call(message.answer_media_group, media=media)
                    await safe_delete_message(status)
                    return

                video_url = data.get("hdplay") or data.get("play") or data.get("play_addr")
                if not video_url:
                    raise RuntimeError("TikWM не вернул ссылку на видео")

                await tg_call(
                    message.answer_video,
                    video=video_url,
                    caption=title[:1024] if title else None,
                    supports_streaming=True,
                )
                await safe_delete_message(status)
                return

        if is_instagram(url):
            result = await with_retry(download_instagram_apify, url)
            temp_dirs.append(result["temp_dir"])
            stat_inc("media_total")
            stat_inc("instagram_downloads")
            async with get_chat_lock(message.chat.id):
                await safe_status_edit(status, "Отправляю...")
                await send_local_media(message, result["files"], result.get("caption"))
                await safe_delete_message(status)
                return

        if is_twitter(url):
            result = await with_retry(download_twitter_fx, url)
            temp_dirs.append(result["temp_dir"])
            stat_inc("media_total")
            stat_inc("twitter_downloads")
            async with get_chat_lock(message.chat.id):
                await safe_status_edit(status, "Отправляю...")
                await send_local_media(message, result["files"], result.get("caption"))
                await safe_delete_message(status)
                return

        if is_direct_image(url):
            result = await with_retry(download_direct_image, url)
            temp_dirs.append(result["temp_dir"])
            stat_inc("media_total")
            stat_inc("direct_image_downloads")
            async with get_chat_lock(message.chat.id):
                await safe_status_edit(status, "Отправляю...")
                await send_local_media(message, result["files"], result.get("caption"))
                await safe_delete_message(status)
                return

        stat_inc("media_total")
        stat_inc("ytdlp_downloads")
        result = await with_retry(download_ytdlp, url)
        temp_dirs.append(result["temp_dir"])

        async with get_chat_lock(message.chat.id):
            await safe_status_edit(status, "Отправляю...")
            await send_local_media(message, result["files"], result.get("caption"))
            await safe_delete_message(status)

    except aiohttp.ClientResponseError as e:
        stat_inc("media_errors")
        text = str(e).lower()
        if is_tiktok(url):
            await safe_status_edit(status, "Хозяин, TikTok не отдал ничего... Попробуй ещё раз попозже, лапочка (⁠｡⁠・⁠/⁠/⁠ε⁠/⁠/⁠・⁠｡⁠)")
        elif is_instagram(url):
            await safe_status_edit(status, "Хозяин, Instagram почему-то не отдал ничего, Попробуй ещё разочек (⁠｡⁠・⁠/⁠/⁠ε⁠/⁠/⁠・⁠｡⁠)")
        elif is_twitter(url):
            if "404" in text:
                await safe_status_edit(status, "Хозяин, Twitter/X пост не найден или его уже удалили, прости пожалуйста (⁠´⁠ ⁠.⁠ ⁠.̫⁠ ⁠.⁠ ⁠`⁠)")
            else:
                await safe_status_edit(status, "Хозяин, Twitter/X почему-то не отдал медиа. Попробуй чуть позже ^^")
        else:
            await safe_status_edit(status, "Упс, не получилось скачать (⁠´⁠ ⁠.⁠ ⁠.̫⁠ ⁠.⁠ ⁠`⁠) Не наказывай меня, Хозяин, но я не знаю почему")

    except asyncio.TimeoutError:
        await safe_status_edit(status, "Хозяин... сервер отвечает слишком долго... п-попробуйте ещё раз, пожалуйста... (つ﹏<。)")

    except DownloadError as e:
        await safe_status_edit(status, human_ytdlp_error(e))

    except TelegramBadRequest:
        await safe_status_edit(status, "Хозяин, я все ещё хороший мальчик, но телеграм не дает отправить это видео (⁠눈⁠‸⁠눈⁠)")

    except TelegramRetryAfter as e:
        await asyncio.sleep(float(e.retry_after) + 1)
        await safe_status_edit(status, "Хозяин... Telegram попросил меня подождать немножко, попробуй ещё разочек ^^")

    except Exception as e:
        print(f"[media] Unhandled error for {url}: {e}")
        if is_instagram(url):
            await safe_status_edit(status, human_instagram_api_error(e))
        elif is_twitter(url):
            await safe_status_edit(status, human_twitter_error(e))
        else:
            await safe_status_edit(status, "Хозяин... простите, пожалуйста... при обработке ссылки что-то пошло не так... TᴖT")

    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = (
        "🐾 Cattemis Bot\n\n"
        "Я умею скачивать медиа из TikTok, Instagram, X/Twitter, "
        "YouTube, Vimeo и прямых ссылок на фото/видео.\n\n"
        "Команды:\n"
        "/help — показать это сообщение\n"
        "/art — отправить случайный арт\n"
        "/artist <id> — отправить арт по id художника\n"
        "/say_cattemis <text> — повторить текст от имени бота\n\n"
        "Просто отправь мне ссылку на фото или видео, и я попробую скачать её."
    )

    await tg_call(message.answer, help_text)

@dp.message(Command("say_cattemis"))
async def cmd_say(message: Message):
    stat_inc("commands_used")
    stat_track_chat(message.chat.id)
    if not await can_use_say(message):
        return

    raw_text = (message.text or "").strip()
    payload = raw_text.partition(" ")[2].strip()

    if not payload and message.reply_to_message:
        payload = (message.reply_to_message.text or message.reply_to_message.caption or "").strip()

    if not payload:
        await tg_call(message.answer, "Использование: /say_cattemis текст\nИли ответь на сообщение командой /say_cattemis")
        return

    async with get_chat_lock(message.chat.id):
        await tg_call(message.answer, payload)

    await safe_delete_message(message)

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    stat_inc("commands_used")
    stat_track_chat(message.chat.id)

    uptime = format_uptime(time.time() - _stats["started_at"])
    chats_count = len(_stats["unique_chats"])

    text = (
        f"Статистика бота:\n"
        f"Uptime: {uptime}\n"
        f"Уникальных чатов: {chats_count}\n"
        f"Сообщений обработано: {_stats['messages_total']}\n"
        f"Команд использовано: {_stats['commands_used']}\n"
        f"LLM вызовов: {_stats['llm_calls']}\n"
        f"LLM ошибок: {_stats['llm_errors']}\n"
        f"Медиа всего: {_stats['media_total']}\n"
        f"TikTok: {_stats['tiktok_downloads']}\n"
        f"Instagram: {_stats['instagram_downloads']}\n"
        f"Twitter/X: {_stats['twitter_downloads']}\n"
        f"Direct image: {_stats['direct_image_downloads']}\n"
        f"yt-dlp: {_stats['ytdlp_downloads']}\n"
        f"Ошибок медиа: {_stats['media_errors']}"
    )

    await tg_call(
        message.answer,
        text,
        reply_parameters=ReplyParameters(message_id=message.message_id),
        parse_mode=None,
    )

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    stat_inc("commands_used")
    stat_track_chat(message.chat.id)
    clear_chat_history(message.chat.id)
    await tg_call(
        message.answer,
        "Память диалога для этого чата очищена, мяу~",
        reply_parameters=ReplyParameters(message_id=message.message_id),
        parse_mode=None,
    )


@dp.message(Command("gamble_cattemis"))
@dp.message(Command("art"))
async def cmd_art(message: Message):
    stat_inc("commands_used")
    stat_track_chat(message.chat.id)
    print(f"[art] command from chat={message.chat.id}")

    link = random_artist_link()
    if not link:
        await tg_call(message.answer, "Хозяин, artists.json пустой или все художники выключены...")
        return

    await process_media_url(message, link.url, initial_status_text=f"Скачиваю артик от {link.label}...")


@dp.message(Command("artist"))
async def cmd_artist(message: Message):
    stat_inc("commands_used")
    stat_track_chat(message.chat.id)
    raw_text = (message.text or "").strip()
    artist_id = raw_text.partition(" ")[2].strip()

    if not artist_id:
        await tg_call(message.answer, "Использование: /artist <id>")
        return

    link = random_artist_link(artist_id)
    if not link:
        await tg_call(message.answer, f"Хозяин, для artist_id='{artist_id}' ничего не найдено.")
        return

    await process_media_url(message, link.url, initial_status_text=f"Скачиваю артик от {link.label}...")


@dp.message()
async def handle_link(message: Message):
    stat_inc("messages_total")
    stat_track_chat(message.chat.id)
    deleted, urls = await moderate_links(message)
    if deleted:
        return

    raw_text = (message.text or message.caption or "").strip()

    if raw_text.startswith("/"):
        return

    if await is_praise_for_bot(message):
        await tg_call(message.answer, random.choice(PRAISE_REPLIES))
        return

    if urls:
        allowed_urls = [url for url in urls if is_allowed_media_link(url)]
        if not allowed_urls:
            if message.chat.type == "private":
                await tg_call(message.answer, "Пришли мне ссылку на фото или видео.")
            return

        await process_media_url(message, allowed_urls[0], initial_status_text="Скачиваю...")
        return

    if not raw_text:
        return

    should_use_llm = False
    if LLM_ENABLED:
        if message.chat.type == "private":
            should_use_llm = True
        elif await is_reply_to_this_bot(message) or await is_bot_mentioned(message):
            should_use_llm = True

    if should_use_llm:
        try:
            async with ChatActionSender.typing(
                bot=bot,
                chat_id=message.chat.id,
                message_thread_id=message.message_thread_id,
            ):
                stat_inc("llm_calls")
                reply = await ask_llm(
                    message.chat.id,
                    raw_text,
                    user_name=message.from_user.first_name if message.from_user else None,
                )

            reply = reply.strip()[:4000] or "..."

            await tg_call(
                message.answer,
                reply,
                reply_parameters=ReplyParameters(message_id=message.message_id),
                parse_mode=None,
            )
        except Exception as e:
            stat_inc("llm_errors")
            print(f"[llm] error: {e}")
            await tg_call(
                message.answer,
                "Хозяин... я задумался слишком сильно и не смог ответить TᴖT",
                reply_parameters=ReplyParameters(message_id=message.message_id),
                parse_mode=None,
            )
        return

    if message.chat.type == "private":
        await tg_call(message.answer, "Пришли мне ссылку на фото или видео.")


async def main():
    banner = """
    ┌──────────────────────────────────────────────┐
│   🐾 Cattemis bot started! Meow meow meow   
└──────────────────────────────────────────────┘
"""
    print(banner.strip())

    load_artists_config()

    if LLM_ENABLED:
        print(f"☆ LLM enabled, base_url={LLM_BASE_URL}, model={LLM_MODEL}")
    else:
        print("☆ LLM disabled")

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())