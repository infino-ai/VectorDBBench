from pydantic import BaseModel, SecretStr

from vectordb_bench.backend.clients.api import DBCaseConfig, DBConfig, MetricType

# Infino distance metrics; all are distances where smaller means nearer.
_METRIC_MAP = {
    MetricType.COSINE: "cosine",
    MetricType.L2: "l2sq",
    MetricType.IP: "negdot",
}

# The hosted platform's default public API endpoint (the split-origin gateway
# host, not the console site). Overridable for staging / a self-hosted deployment.
_DEFAULT_HOST = "https://api.platform.infino.ws"


class InfinoCloudConfig(DBConfig):
    """Connection config for the managed Infino Cloud (Tier-3) gateway.

    The API key is the only secret; it is supplied at runtime (never committed)
    and carried on `Authorization: Bearer`. `host`/`database`/`table_name` are
    plain defaulted names. The database is created-if-absent and reused across
    runs (never deleted); `drop_old` recreates only the table.
    """

    host: str = _DEFAULT_HOST
    api_key: SecretStr
    database: str = "vdbbench"
    table_name: str = "vdbbench_infino"

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "api_key": self.api_key.get_secret_value() if self.api_key else "",
            "database": self.database,
            "table_name": self.table_name,
        }


class InfinoCloudIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type not in _METRIC_MAP:
            msg = f"Infino Cloud does not support metric {self.metric_type}"
            raise ValueError(msg)
        return _METRIC_MAP[self.metric_type]

    def index_param(self) -> dict:
        return {"metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {}
