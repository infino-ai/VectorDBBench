import tempfile

import numpy as np
import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.infino.config import InfinoFTSConfig, InfinoIndexConfig
from vectordb_bench.backend.filter import IntFilter, LabelFilter


class TestInfino:
    def test_case_config_cls_selects_fts(self):
        # The FTS CLI/assembler path resolves the case config via DB.case_config_cls;
        # FTS index_type must yield the FTS config, everything else the vector config.
        assert DB.Infino.case_config_cls(IndexType.FTS) is InfinoFTSConfig
        assert DB.Infino.case_config_cls() is InfinoIndexConfig

    def test_search_mode_config(self):
        # Default is the resident-HNSW path; ivf is opt-out; bad values reject.
        # search_mode is bridged to the engine config file, so it must NOT leak
        # into index_param() (which feeds IndexSpec).
        assert InfinoIndexConfig().search_mode == "hnsw_ivf"
        cfg = InfinoIndexConfig(metric_type=MetricType.COSINE, search_mode="ivf")
        assert cfg.search_mode == "ivf"
        assert "search_mode" not in cfg.index_param()
        with pytest.raises(ValueError, match="search_mode"):
            InfinoIndexConfig(search_mode="bogus")

    def test_ef_case_config(self):
        # ef defaults to 0 (serve at the graph's stamped k->ef curve); a positive
        # value is a fixed serve-time beam; negative is rejected. Like search_mode
        # it is bridged to the engine config, not index_param().
        assert InfinoIndexConfig().ef == 0
        assert InfinoIndexConfig(ef=768).ef == 768
        assert "ef" not in InfinoIndexConfig(metric_type=MetricType.COSINE, ef=768).index_param()
        with pytest.raises(ValueError, match="ef"):
            InfinoIndexConfig(ef=-1)

    def test_ef_search_config_bridging(self, tmp_path):
        # search_mode + ef are bridged to the engine's YAML config; only values
        # that diverge from the engine default (ivf + stamped curve) are written.
        pytest.importorskip("infino")
        from vectordb_bench.backend.clients.infino.infino import Infino

        cfg_path = tmp_path / "_infino_engine_cfg" / "infino" / "config.yaml"

        def written(search_mode: str, ef: int) -> str:
            client = object.__new__(Infino)  # bypass __init__: no engine connect
            client._search_mode = search_mode
            client._ef = ef
            client.data_path = str(tmp_path)
            client._local = tmp_path  # local scratch (== data_path for a local store)
            if cfg_path.exists():
                cfg_path.unlink()
            client._apply_search_mode_config()
            return cfg_path.read_text() if cfg_path.exists() else ""

        both = written("hnsw_ivf", 768)
        assert "search_mode: hnsw_ivf" in both
        assert "hnsw_ef_search: 768" in both

        curve = written("hnsw_ivf", 0)
        assert "search_mode: hnsw_ivf" in curve
        assert "hnsw_ef_search" not in curve

        assert written("ivf", 0) == ""  # pure engine default: nothing written

    def test_insert_and_search(self):
        assert DB.Infino.value == "Infino"

        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls
        case_config_cls = DB.Infino.case_config_cls()

        dim = 16
        count = 2_000
        rng = np.random.default_rng(0)
        embeddings = rng.random((count, dim)).tolist()

        with tempfile.TemporaryDirectory() as data_path:
            db_config = config_cls(data_path=data_path).to_dict()
            # 2K rows sit inside the engine's default rerank budget, so the
            # engine-decided serving is exact for the assertion.
            db_case_config = case_config_cls(metric_type=MetricType.L2)

            client = dbcls(
                dim=dim,
                db_config=db_config,
                db_case_config=db_case_config,
                collection_name="test_infino",
                drop_old=True,
            )

            with client.init():
                inserted, err = client.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
                assert err is None
                assert inserted == count

            with client.init():
                test_id = 42
                res = client.search_embedding(query=embeddings[test_id], k=10)
                assert res[0] == test_id, f"nearest neighbor id {res[0]} != query id {test_id}"

    def test_insert_buffering_persists_every_row(self, monkeypatch: pytest.MonkeyPatch):
        # insert_embeddings buffers fed rows and commits large appends. Rows must
        # survive both the mid-load threshold flush and the init()-exit flush of
        # the sub-threshold remainder (the case where the corpus < _FLUSH_ROWS).
        pytest.importorskip("infino")  # engine binding; skip if the extra isn't installed
        monkeypatch.setattr("vectordb_bench.backend.clients.infino.infino._FLUSH_ROWS", 50)

        dim = 16  # engine requires dim in [16, 4096]
        count = 130  # 20-row feeds flush at 60 twice (120 rows), leaving a 10-row remainder
        rng = np.random.default_rng(1)
        embeddings = rng.random((count, dim)).tolist()

        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls
        case_config_cls = DB.Infino.case_config_cls()

        with tempfile.TemporaryDirectory() as data_path:
            client = dbcls(
                dim=dim,
                db_config=config_cls(data_path=data_path).to_dict(),
                db_case_config=case_config_cls(metric_type=MetricType.L2),
                collection_name="test_infino_buffer",
                drop_old=True,
            )
            with client.init():
                total = 0
                for start in range(0, count, 20):
                    chunk = embeddings[start : start + 20]
                    inserted, err = client.insert_embeddings(
                        embeddings=chunk, metadata=list(range(start, start + len(chunk)))
                    )
                    assert err is None
                    total += inserted
                assert total == count
            # The 10-row remainder was flushed at init() exit; a row from it
            # (id 129, the last inserted) must be searchable.
            with client.init():
                res = client.search_embedding(query=embeddings[129], k=1)
                assert res[0] == 129, f"remainder row not persisted: got {res[0]}"

    def test_numge_filter(self):
        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls
        case_config_cls = DB.Infino.case_config_cls()

        dim = 16
        count = 1_000
        threshold = 500
        rng = np.random.default_rng(1)
        embeddings = rng.random((count, dim)).tolist()

        with tempfile.TemporaryDirectory() as data_path:
            client = dbcls(
                dim=dim,
                db_config=config_cls(data_path=data_path).to_dict(),
                db_case_config=case_config_cls(metric_type=MetricType.L2),
                collection_name="test_numge",
                drop_old=True,
            )
            with client.init():
                client.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))

            with client.init():
                client.prepare_filter(IntFilter(filter_rate=0.5, int_field="id", int_value=threshold))
                query_id = 700
                res = client.search_embedding(query=embeddings[query_id], k=10)
                assert res[0] == query_id
                assert all(r >= threshold for r in res), f"NumGE leaked ids < {threshold}: {res}"

    def test_strequal_filter(self):
        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls
        case_config_cls = DB.Infino.case_config_cls()

        dim = 16
        count = 1_000
        rng = np.random.default_rng(2)
        embeddings = rng.random((count, dim)).tolist()
        label_filter = LabelFilter(label_percentage=0.5)
        target = label_filter.label_value
        # Even ids carry the target label; odd ids get a different one.
        labels = [target if i % 2 == 0 else "label_other" for i in range(count)]

        with tempfile.TemporaryDirectory() as data_path:
            client = dbcls(
                dim=dim,
                db_config=config_cls(data_path=data_path).to_dict(),
                db_case_config=case_config_cls(metric_type=MetricType.L2),
                collection_name="test_strequal",
                drop_old=True,
                with_scalar_labels=True,
            )
            with client.init():
                client.insert_embeddings(
                    embeddings=embeddings,
                    metadata=list(range(count)),
                    labels_data=labels,
                )

            with client.init():
                client.prepare_filter(label_filter)
                query_id = 200  # even -> carries target label
                res = client.search_embedding(query=embeddings[query_id], k=10)
                assert res[0] == query_id
                assert all(r % 2 == 0 for r in res), f"StrEqual leaked non-target rows: {res}"

    def test_fts_bm25(self):
        assert DB.Infino.init_cls.supports_full_text_search() is True

        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls

        docs = ["alpha beta", "gamma delta", "beta gamma", "unique zebra sentence"]
        doc_ids = [str(i) for i in range(len(docs))]

        with tempfile.TemporaryDirectory() as data_path:
            client = dbcls(
                dim=0,
                db_config=config_cls(data_path=data_path).to_dict(),
                db_case_config=InfinoFTSConfig(),
                collection_name="test_fts",
                drop_old=True,
            )
            with client.init():
                inserted, err = client.insert_documents(texts=docs, doc_ids=doc_ids)
                assert err is None
                assert inserted == len(docs)

            with client.init():
                res = client.search_documents(query="zebra", k=10)
                assert res == ["3"], f"expected doc 3 for 'zebra', got {res}"
                # returned ids are strings matching the FTS ground-truth dtype
                assert all(isinstance(r, str) for r in res)
