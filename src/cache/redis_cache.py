"""Quản lý shared async Redis client với cache fail-open.

Không kết nối khi cache bị tắt hoặc dependency không có. Lỗi Redis trả ``None``
để caller tiếp tục không cache; module không tự xóa key hay thay cache identity.
"""
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

async def get_redis() -> Optional['redis.Redis']:
    """Trả async Redis client khi cache được bật và kết nối thành công."""
    global _redis_client
    
    if not REDIS_AVAILABLE:
        return None
        
    cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    if not cache_enabled:
        return None
        
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            # Ping ngay khi tạo để không phát tán client chưa usable cho caller.
            await _redis_client.ping()
        except Exception as e:
            logger.warning(
                "Could not connect to Redis: %s. Cache will be disabled.",
                sanitize_fallback_reason(e),
            )
            _redis_client = None
            
    return _redis_client

async def close_redis():
    """Đóng shared Redis connection nếu đã được tạo."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.close()
        except Exception as e:
            logger.warning("Error closing Redis: %s", sanitize_fallback_reason(e))
        _redis_client = None

async def ping_redis() -> bool:
    """Kiểm Redis reachable mà không thay đổi dữ liệu."""
    client = await get_redis()
    if client is None:
        return False
    try:
        return await client.ping()
    except Exception:
        return False
