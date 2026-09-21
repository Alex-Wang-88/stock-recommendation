"""串行节流的 HTTP 客户端。

为什么需要它（实测教训）：东财的风控是 **「端点 + 频率」共同触发**，不是单纯封 IP。
实测在同一秒内 `push2his /api/qt/stock/fflow/daykline/get` 成功（120 条），
而 `push2his /api/qt/stock/kline/get`、`push2 /api/qt/clist/get` 直接被
`RemoteDisconnected` 掐断。⇒ **一个端点失败，不能推断成"IP 被封"**。

因此本模块提供两件事：
1. 所有请求串行 + 最小间隔 + 随机抖动 + 会话复用（把频率压到风控阈值以下）。
2. `probe()`：用一个**已知可用**的对照端点判断"到底是被封了，还是这个端点/参数不对"。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
)

# 已知可用的对照端点：东财个股资金流日线（本轮实测稳定返回）。
CONTROL_PROBE_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
CONTROL_PROBE_PARAMS = {
    "secid": "1.600519",
    "fields1": "f1,f2,f3,f7",
    "fields2": "f51,f52,f53,f54,f55,f56,f57",
    "lmt": "5",
}


class FetchError(RuntimeError):
    """请求在重试耗尽后仍然失败。"""


@dataclass
class ThrottledClient:
    min_interval: float = 1.2
    jitter: float = 0.4
    retries: int = 3
    backoff: float = 2.0
    timeout: float = 15.0
    user_agent: str = DEFAULT_UA
    session: requests.Session | None = field(default=None, repr=False)
    last_call: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    # -- 节流 --------------------------------------------------------------- #

    def _wait(self) -> None:
        """把两次请求的间隔压到 min_interval 以上，再加随机抖动。"""

        elapsed = time.time() - self.last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining + random.uniform(0.0, self.jitter))

    # -- 请求 --------------------------------------------------------------- #

    def get(
        self, url: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> requests.Response:
        assert self.session is not None
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout, **kwargs)
                self.last_call = time.time()
                if response.status_code >= 400:
                    raise FetchError(f"HTTP {response.status_code} for {url}")
                return response
            except Exception as exc:  # noqa: BLE001
                self.last_call = time.time()
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
        raise FetchError(f"{url} 请求失败（{self.retries} 次重试）：{last_error}") from last_error

    def get_json(self, url: str, params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.get(url, params=params, **kwargs).json()

    # -- 诊断 --------------------------------------------------------------- #

    def probe(self) -> bool:
        """用对照端点判断网络是否整体不可用。

        返回 True ⇒ 通路正常。此时若业务端点仍失败，**是端点或参数的问题，
        不是被封 IP**，不要盲目加冷却时间。
        """

        try:
            payload = self.get_json(CONTROL_PROBE_URL, params=dict(CONTROL_PROBE_PARAMS))
        except FetchError:
            return False
        return bool((payload.get("data") or {}).get("klines"))
