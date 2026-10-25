"""Quản lý shared async Redis client với cache fail-open.

Không kết nối khi cache bị tắt hoặc dependency không có. Lỗi Redis trả ``None``
để caller tiếp tục không cache; module không tự xóa key hay thay cache identity.
"""
import asyncio
import os
import logging
from typing import Optional

from src.quality.safe_fallback import sanitize_fallback_reason

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Một shared client cho process; application shutdown gọi ``close_redis``.
_redis_client: Optional['redis.Redis'] = None
_redis_lock: asyncio.Lock | None = None


def _client_lock() -> asyncio.Lock:
    global _redis_lock
    if _redis_lock is None:
        _redis_lock = asyncio.Lock()
    return _redis_lock


async def _close_client(client) -> None:
    try:
        await client.aclose()
    except Exception as exc:
        logger.warning("Error closing Redis: %s", sanitize_fallback_reason(exc))

async def get_redis() -> Optional['redis.Redis']:
    """Trả async Redis client khi cache được bật và kết nối thành công."""
    global _redis_client
    
    if not REDIS_AVAILABLE:
        return None
        
    cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    if not cache_enabled:
        return None
        
    if _redis_client is None:
        async with _client_lock():
            if _redis_client is None:
                client = None
                try:
                    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                    client = redis.from_url(redis_url, decode_responses=True)
                    # Publish only after readiness; concurrent callers wait here.
                    await client.ping()
                    _redis_client = client
                except Exception as e:
                    logger.warning(
                        "Could not connect to Redis: %s. Cache will be disabled.",
                        sanitize_fallback_reason(e),
                    )
                finally:
                    if client is not None and _redis_client is not client:
                        await _close_client(client)
            
    return _redis_client

async def close_redis():
    """Đóng shared Redis connection nếu đã được tạo."""
    global _redis_client
    async with _client_lock():
        client, _redis_client = _redis_client, None
        if client is not None:
            await _close_client(client)

async def ping_redis() -> bool:
    """Kiểm Redis reachable mà không thay đổi dữ liệu."""
    client = await get_redis()
    if client is None:
        return False
    try:
        return await client.ping()
    except Exception:
        return False
