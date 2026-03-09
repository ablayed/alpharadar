from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

from src.data.base import BaseDataProvider, SECFiling
from src.storage.database import Database
from src.utils.rate_limiter import RateLimitedClient

logger = logging.getLogger(__name__)

STOCK_CIK_MAP: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "AMZN": "0001018724",
    "AMD": "0000002488",
    "TSLA": "0001318605",
    "ARM": "0001973239",
    "PLTR": "0001321655",
}

EIGHT_K_ITEMS: dict[str, str] = {
    "1.01": "Entry into a Material Agreement",
    "1.02": "Termination of a Material Agreement",
    "2.01": "Completion of Acquisition/Disposition",
    "2.02": "Results of Operations and Financial Condition",
    "2.05": "Costs for Exit/Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Delisting or Transfer Failure",
    "4.01": "Changes in Certifying Accountant",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Appointment of Directors/Officers",
    "5.03": "Amendments to Articles",
    "5.07": "Shareholder Vote Submission",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


class CompanyTickersEntry(BaseModel):
    cik_str: int
    ticker: str
    title: str


class RecentFilings(BaseModel):
    accessionNumber: list[str] = Field(default_factory=list)
    filingDate: list[str] = Field(default_factory=list)
    form: list[str] = Field(default_factory=list)
    primaryDocument: list[str] = Field(default_factory=list)
    primaryDocDescription: list[str] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)


class FilingsWrapper(BaseModel):
    recent: RecentFilings


class SubmissionsResponse(BaseModel):
    name: str | None = None
    filings: FilingsWrapper


class EdgarClient(BaseDataProvider):
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_BASE = "https://data.sec.gov/submissions"

    def __init__(
        self,
        db: Database,
        rps: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.db = db
        self._client: httpx.AsyncClient | None = None
        self._rate_client: RateLimitedClient | None = None
        self._limiter = AsyncLimiter(max_rate=rps, time_period=1)
        self._transport = transport

    async def initialize(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            timeout=20.0,
            transport=self._transport,
            headers={
                "User-Agent": "AlphaRadar contact@alpharadar.dev",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        self._rate_client = RateLimitedClient(self._client, self._limiter, "edgar")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._rate_client = None

    async def health_check(self) -> bool:
        try:
            cik = await self.lookup_cik("AAPL")
            return cik is not None
        except Exception:
            logger.exception("EDGAR health check failed")
            return False

    async def lookup_cik(self, symbol: str) -> str | None:
        upper = symbol.upper()
        if upper in STOCK_CIK_MAP:
            return STOCK_CIK_MAP[upper]

        cache_key = "edgar:company_tickers"
        payload = await self._request_json(url=self.TICKERS_URL, cache_key=cache_key, ttl=86_400)
        if not isinstance(payload, dict):
            return None
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            ticker = entry.get("ticker")
            if isinstance(ticker, str) and ticker.upper() == upper:
                parsed = CompanyTickersEntry.model_validate(entry)
                return str(parsed.cik_str).zfill(10)
        return None

    async def get_recent_8k_filings(self, symbols: list[str], days_back: int = 1) -> list[SECFiling]:
        filings: list[SECFiling] = []
        cutoff = date.today() - timedelta(days=days_back)
        cutoff_str = cutoff.isoformat()

        for symbol in symbols:
            cik = await self.lookup_cik(symbol)
            if cik is None:
                logger.warning("Missing CIK for %s, skipping EDGAR lookup", symbol)
                continue

            padded_cik = cik.zfill(10)
            cache_key = f"edgar:submissions:{padded_cik}"
            payload = await self._request_json(
                url=f"{self.SUBMISSIONS_BASE}/CIK{padded_cik}.json",
                cache_key=cache_key,
                ttl=3_600,
            )
            if not isinstance(payload, dict):
                continue
            parsed = SubmissionsResponse.model_validate(payload)
            recent = parsed.filings.recent
            total = len(recent.form)
            for i in range(total):
                form = recent.form[i] if i < len(recent.form) else ""
                filing_date = recent.filingDate[i] if i < len(recent.filingDate) else ""
                if form != "8-K" or filing_date < cutoff_str:
                    continue

                accession = recent.accessionNumber[i] if i < len(recent.accessionNumber) else ""
                primary_doc = recent.primaryDocument[i] if i < len(recent.primaryDocument) else ""
                description = (
                    recent.primaryDocDescription[i]
                    if i < len(recent.primaryDocDescription) and recent.primaryDocDescription[i]
                    else ""
                )
                item_codes = recent.items[i] if i < len(recent.items) else ""
                if not description and item_codes:
                    first_item = item_codes.split(",")[0].strip()
                    description = EIGHT_K_ITEMS.get(first_item, "8-K Filing")
                if not description:
                    description = "8-K Filing"

                accession_clean = accession.replace("-", "")
                archives_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(padded_cik)}/{accession_clean}/{primary_doc}"
                )
                filings.append(
                    SECFiling(
                        company_name=parsed.name or symbol.upper(),
                        ticker=symbol.upper(),
                        form_type=form,
                        filing_date=filing_date,
                        description=description,
                        url=archives_url,
                    )
                )
        return filings

    async def _request_json(self, url: str, cache_key: str, ttl: int) -> Any:
        cached = await self.db.get_cached(cache_key)
        if cached is not None:
            return json.loads(cached)

        if self._rate_client is None:
            raise RuntimeError("EdgarClient is not initialized")

        response = await self._rate_client.get(url)
        payload: Any = response.json()
        await self.db.set_cached(cache_key, json.dumps(payload), ttl_seconds=ttl)
        return payload

