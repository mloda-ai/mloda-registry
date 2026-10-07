"""Tests for load_dataset: OpenLineage dataset naming of data loads."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestOpenLineageLoadDatasetNaming:
    """load_dataset maps an identity to an OpenLineage (namespace, name), else (fallback_namespace, identity)."""

    # "{tmp}" is replaced by an existing directory under tmp_path.
    @pytest.mark.parametrize(
        ("identity", "expected"),
        [
            pytest.param("s3://bucket/key.parquet", ("s3://bucket", "key.parquet"), id="s3"),
            pytest.param("s3://bucket/a/b/key.parquet", ("s3://bucket", "a/b/key.parquet"), id="s3_nested_key"),
            pytest.param("S3://bucket/key", ("s3://bucket", "key"), id="uppercase_scheme"),
            pytest.param("gs://bucket/key.csv", ("gs://bucket", "key.csv"), id="gcs"),
            pytest.param("s3://bucket", ("fb", "s3://bucket"), id="s3_bucket_only"),
            pytest.param("s3://bucket/", ("fb", "s3://bucket/"), id="s3_empty_key"),
            pytest.param("s3:///key", ("fb", "s3:///key"), id="s3_empty_bucket"),
            pytest.param("s3://bucket:9000/k", ("fb", "s3://bucket:9000/k"), id="s3_port"),
            pytest.param("gs://bucket", ("fb", "gs://bucket"), id="gcs_bucket_only"),
            pytest.param("s3://bucket//key", ("s3://bucket", "/key"), id="s3_double_slash_drops_one"),
            pytest.param("s3://bucket/key?versionId=1", ("fb", "s3://bucket/key?versionId=1"), id="query"),
            pytest.param("s3://bucket/key#frag", ("fb", "s3://bucket/key#frag"), id="fragment"),
            pytest.param("s3://user@bucket/key", ("fb", "s3://user@bucket/key"), id="non_abfss_at_sign"),
            pytest.param(
                "abfss://raw@acct.dfs.core.windows.net/p/f.parquet",
                ("abfss://raw@acct.dfs.core.windows.net", "p/f.parquet"),
                id="abfss",
            ),
            pytest.param(
                "abfss://acct.dfs.core.windows.net/p",
                ("fb", "abfss://acct.dfs.core.windows.net/p"),
                id="abfss_without_container",
            ),
            pytest.param("abfss://raw@example.com/p", ("fb", "abfss://raw@example.com/p"), id="abfss_on_another_host"),
            pytest.param(
                "abfss://raw@acct.dfs.core.windows.net:443/p",
                ("fb", "abfss://raw@acct.dfs.core.windows.net:443/p"),
                id="abfss_port",
            ),
            pytest.param(
                "abfss://raw@acct.dfs.core.windows.net",
                ("fb", "abfss://raw@acct.dfs.core.windows.net"),
                id="abfss_empty_path",
            ),
            pytest.param("file:///abs/path.csv", ("file", "/abs/path.csv"), id="file_uri"),
            pytest.param("file://host/x", ("fb", "file://host/x"), id="file_uri_with_host"),
            pytest.param("{tmp}/data.csv", ("file", "{tmp}/data.csv"), id="existing_absolute_path"),
            pytest.param("/definitely/missing/path.csv", ("file", "/definitely/missing/path.csv"), id="missing_path"),
            pytest.param("file:///", ("fb", "file:///"), id="file_uri_root"),
            pytest.param("/", ("fb", "/"), id="root_path"),
            pytest.param("rel/data.csv", ("fb", "rel/data.csv"), id="relative_path"),
            pytest.param("/x/db.sqlite::t", ("fb", "/x/db.sqlite::t"), id="sqlite_table"),
            pytest.param("jdbc:postgresql://h:5432", ("fb", "jdbc:postgresql://h:5432"), id="jdbc"),
            pytest.param("https://h/p", ("fb", "https://h/p"), id="https"),
            pytest.param("hdfs://nn/p", ("fb", "hdfs://nn/p"), id="hdfs"),
            pytest.param("s3a://bucket/key", ("fb", "s3a://bucket/key"), id="s3a"),
            pytest.param("str", ("fb", "str"), id="type_name"),
            pytest.param("{a, b}", ("fb", "{a, b}"), id="set_form"),
            pytest.param("C:\\data\\x.csv", ("fb", "C:\\data\\x.csv"), id="windows_path"),
        ],
    )
    def test_load_dataset_maps_or_falls_back_without_raising(
        self, identity: str, expected: tuple[str, str], tmp_path: Path
    ) -> None:
        from mloda.community.extenders.openlineage.dataset_naming import load_dataset

        (tmp_path / "data.csv").write_text("a\n1\n")
        identity = identity.replace("{tmp}", str(tmp_path))
        want = (expected[0], expected[1].replace("{tmp}", str(tmp_path)))

        assert load_dataset(identity, "fb") == want
