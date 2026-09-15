"""Managed Infino Cloud (Tier-3) VDBBench client.

Drives the hosted platform gateway's public REST API (bearer-auth data plane):
create-if-absent database + table, Arrow-IPC ingest, synchronous force-optimize,
and vector search. Mirrors the embedded `infino` client's lifecycle so the
managed-service curve is comparable to the self-hosted series.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

import pyarrow as pa
import requests

from ..api import VectorDB
from .config import InfinoCloudIndexConfig

log = logging.getLogger(__name__)

_ARROW_STREAM = "application/vnd.apache.arrow.stream"
_ID_FIELD = "id"
_VECTOR_FIELD = "emb"
# Long read timeout: optimize is synchronous (the POST returns only when the
# drain+build+calibrate finishes), which can take many minutes at scale.
_OPTIMIZE_TIMEOUT = 3 * 60 * 60
_REQUEST_TIMEOUT = 600
# The gateway rejects request bodies over 5 MiB (it drops the TLS connection
# mid-upload). Each append is chunked to sit safely under that, computed from
# the vector dim; the append count — hence delta-fragment count before the
# force-optimize — is governed by this limit, not by the bench's batch size.
_MAX_APPEND_BODY_BYTES = 4 * 1024 * 1024  # headroom under the gateway's 5 MiB
_APPEND_RETRIES = 8  # absorb transient WAN blips + occasional in-flight-write 409s
# The gateway takes one write per table at a time: a concurrent write loses with
# 409 and a still-activating worker answers 503, both transient and retryable.
_RETRYABLE_STATUS = (409, 503)
_DEFAULT_RETRY_AFTER = 1.0  # seconds, when the server sends no Retry-After
# Optimize is synchronous server-side, but at scale it outlasts the fronting
# load balancer's idle timeout, which cuts the wait with a 502/503/504 (or a
# dropped socket) while the compaction keeps running on the worker. A cut is not
# failure: re-issue until one returns 200 — on an already-optimized table that is
# a near-noop, the authoritative "done" signal — bounded by _OPTIMIZE_TIMEOUT.
_OPTIMIZE_CUT_STATUS = (502, 503, 504)
_OPTIMIZE_POLL_INTERVAL = 15.0  # seconds between completion re-checks after a cut


class InfinoCloud(VectorDB):
    # Append is single-writer per table (a concurrent write 409s), so the bench
    # must serialize ingest: thread_safe=False clamps the insert runner to one
    # worker. The concurrent search (QPS) phase is multiprocess and unaffected.
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: InfinoCloudIndexConfig,
        collection_name: str = "vdbbench_infino",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.case_config = db_case_config
        self.base = str(db_config["host"]).rstrip("/")
        self._api_key = db_config["api_key"]
        self.database = db_config["database"]
        self.table = db_config.get("table_name") or collection_name
        self.metric = db_case_config.parse_metric()
        self._session: requests.Session | None = None

        if drop_old:
            self._ensure_database()
            self._drop_table()
            self._create_table()

    # ---- auth / transport --------------------------------------------------

    def _headers(self, content_type: str = "application/json") -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Content-Type": content_type,
        }

    def _http(self) -> requests.Session:
        # Init-time control calls run before init(); use a throwaway session then.
        return self._session or requests.Session()

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self._session = requests.Session()
        try:
            yield
        finally:
            self._session.close()
            self._session = None

    # ---- lifecycle (data plane, bearer) ------------------------------------

    def _ensure_database(self):
        """Create the database if absent; reused across runs, never deleted."""
        resp = self._http().post(
            f"{self.base}/v1/databases",
            headers=self._headers(),
            json={"name": self.database},
            timeout=_REQUEST_TIMEOUT,
        )
        # 201 created, or already-exists (409/400) — both fine to proceed on.
        if resp.status_code not in (200, 201, 409):
            if resp.status_code == 400 and "exist" in resp.text.lower():
                return
            resp.raise_for_status()

    def _drop_table(self):
        resp = self._http().post(
            f"{self.base}/v1/drop_table/{self.database}",
            headers=self._headers(),
            json={"table_name": self.table, "purge": True},
            timeout=_REQUEST_TIMEOUT,
        )
        if resp.status_code not in (200, 201, 204, 404):
            log.warning("drop_table (%s): %s %s", self.table, resp.status_code, resp.text[:200])

    def _create_table(self):
        body = {
            "table_name": self.table,
            "schema": [
                {"name": _ID_FIELD, "type": "i64", "nullable": False},
                {"name": _VECTOR_FIELD, "type": "vector", "dim": self.dim, "nullable": False},
            ],
            "indexes": {
                "vector": [{"column": _VECTOR_FIELD, "dim": self.dim, "metric": self.metric}],
            },
        }
        resp = self._http().post(
            f"{self.base}/v1/create_table/{self.database}",
            headers=self._headers(),
            json=body,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

    # ---- ingest ------------------------------------------------------------

    def _encode(self, embeddings: list[list[float]], metadata: list[int]) -> bytes:
        schema = pa.schema(
            [
                pa.field(_ID_FIELD, pa.int64(), nullable=False),
                pa.field(
                    _VECTOR_FIELD,
                    pa.list_(pa.field("item", pa.float32(), nullable=True), self.dim),
                    nullable=False,
                ),
            ]
        )
        flat = [x for row in embeddings for x in row]
        emb = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), self.dim)
        batch = pa.RecordBatch.from_arrays([pa.array(metadata, type=pa.int64()), emb], schema=schema)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
        return sink.getvalue().to_pybytes()

    def _rows_per_chunk(self) -> int:
        """Rows that fit in one append under the gateway's body limit, from the
        vector dim. int64 id (8B) + dim float32 (dim*4B) per row, plus a little
        Arrow framing; the safe budget carries the headroom."""
        per_row = 8 + self.dim * 4
        return max(1, _MAX_APPEND_BODY_BYTES // per_row)

    @staticmethod
    def _retry_after(resp: requests.Response, attempt: int) -> float:
        """Honor the server's Retry-After header; else back off gently."""
        raw = resp.headers.get("Retry-After")
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        return _DEFAULT_RETRY_AFTER * (attempt + 1)

    def _append(self, embeddings: list[list[float]], metadata: list[int]):
        """POST one already-sized chunk. Retries the documented transient
        failures — a 409 (another write to the table was in flight) or a 503
        (workers still activating), honoring Retry-After — and transport blips.
        A terminal 4xx (400/401/404) raises immediately without wasting retries.
        """
        body = self._encode(embeddings, metadata)
        last: Exception | None = None
        for attempt in range(_APPEND_RETRIES):
            try:
                resp = self._http().post(
                    f"{self.base}/v1/append/{self.database}?table={self.table}",
                    headers=self._headers(_ARROW_STREAM),
                    data=body,
                    timeout=_REQUEST_TIMEOUT,
                )
            except requests.RequestException as e:
                # No response (connection reset, TLS EOF, timeout): back off, retry.
                last = e
                if attempt + 1 < _APPEND_RETRIES:
                    time.sleep(_DEFAULT_RETRY_AFTER * (attempt + 1))
                continue
            if resp.status_code in _RETRYABLE_STATUS:
                last = requests.HTTPError(f"{resp.status_code} on append", response=resp)
                if attempt + 1 < _APPEND_RETRIES:
                    time.sleep(self._retry_after(resp, attempt))
                continue
            resp.raise_for_status()  # terminal 4xx/5xx: raise now
            return
        raise last

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        # Split the batch into gateway-sized appends. The batch is all-or-nothing
        # to the caller: on failure return (0, err) so the bench's retry re-sends
        # the whole batch rather than double-appending a partial prefix.
        step = self._rows_per_chunk()
        try:
            for i in range(0, len(metadata), step):
                self._append(embeddings[i : i + step], metadata[i : i + step])
            return len(metadata), None
        except Exception as e:
            log.warning("insert_embeddings failed: %s", e)
            return 0, e

    # ---- optimize (synchronous force) --------------------------------------

    def optimize(self, data_size: int | None = None):
        """Force a table optimize and block until it completes.

        The compaction is synchronous on the worker, but at scale it runs longer
        than the fronting load balancer's idle timeout, which cuts the wait with a
        502/503/504 (or a dropped socket) while the work continues server-side. So
        a cut is not failure — re-issue optimize until one returns 200 (a near-noop
        on an already-optimized table, the authoritative "done" signal), bounded by
        _OPTIMIZE_TIMEOUT.
        """
        deadline = time.monotonic() + _OPTIMIZE_TIMEOUT
        while True:
            try:
                resp = self._http().post(
                    f"{self.base}/v1/optimize/{self.database}",
                    headers=self._headers(),
                    json={"table_name": self.table},
                    timeout=_REQUEST_TIMEOUT,
                )
            except requests.RequestException:
                # Socket dropped while a long compaction runs server-side: poll on.
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_OPTIMIZE_POLL_INTERVAL)
                continue
            if resp.status_code == 200:
                return
            # A load-balancer cut (5xx) or an in-flight-write 409: the compaction
            # is still running server-side — wait, then re-check for completion.
            if resp.status_code in _OPTIMIZE_CUT_STATUS or resp.status_code == 409:
                if time.monotonic() >= deadline:
                    resp.raise_for_status()
                wait = self._retry_after(resp, 0) if resp.status_code == 409 else _OPTIMIZE_POLL_INTERVAL
                time.sleep(wait)
                continue
            resp.raise_for_status()  # a real error (4xx): surface it
            return

    # ---- search ------------------------------------------------------------

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        resp = self._http().post(
            f"{self.base}/v1/vector_search/{self.database}",
            headers=self._headers(),
            json={
                "table_name": self.table,
                "field_name": _VECTOR_FIELD,
                "query": query,
                "k": k,
                "projection": [_ID_FIELD],
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
        rows = rows if isinstance(rows, list) else []
        return [int(r[_ID_FIELD]) for r in rows if _ID_FIELD in r]

    def need_normalize_cosine(self) -> bool:
        return self.metric == "cosine"
