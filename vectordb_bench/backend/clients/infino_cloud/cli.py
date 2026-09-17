from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from .. import DB
from ..api import MetricType


class InfinoCloudTypedDict(TypedDict):
    host: Annotated[
        str,
        click.option(
            "--host", type=str, default="https://api.platform.infino.ws", help="Infino Cloud gateway base URL"
        ),
    ]
    api_key: Annotated[
        str,
        click.option(
            "--api-key",
            type=str,
            envvar="INFINO_CLOUD_API_KEY",
            required=True,
            help="Infino Cloud API key (bearer). Prefer the INFINO_CLOUD_API_KEY env var.",
        ),
    ]
    database: Annotated[
        str, click.option("--database", type=str, default="vdbbench", help="Database name (created if absent, reused)")
    ]
    table_name: Annotated[
        str,
        click.option("--table-name", type=str, default="vdbbench_infino", help="Table name (recreated on drop-old)"),
    ]
    metric: Annotated[
        str,
        click.option(
            "--metric", type=str, default="cosine", help="Distance metric: 'cosine', 'l2'/'l2sq', or 'ip'/'negdot'."
        ),
    ]


class InfinoCloudIndexTypedDict(CommonTypedDict, InfinoCloudTypedDict): ...


def _metric(name: str) -> MetricType | None:
    m = (name or "").lower()
    if m == "cosine":
        return MetricType.COSINE
    if m in ("l2", "l2sq", "euclidean"):
        return MetricType.L2
    if m in ("ip", "negdot", "dot"):
        return MetricType.IP
    return None


@cli.command()
@click_parameter_decorators_from_typed_dict(InfinoCloudIndexTypedDict)
def InfinoCloud(**parameters: Unpack[InfinoCloudIndexTypedDict]):
    from .config import InfinoCloudConfig, InfinoCloudIndexConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.InfinoCloud,
        db_config=InfinoCloudConfig(
            db_label=parameters["db_label"],
            host=parameters["host"],
            api_key=SecretStr(parameters["api_key"]),
            database=parameters["database"],
            table_name=parameters["table_name"],
        ),
        db_case_config=InfinoCloudIndexConfig(metric_type=_metric(parameters["metric"])),
        **parameters,
    )
