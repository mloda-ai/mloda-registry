"""Tests for the shared URI identity sanitizer and the raw-first identity resolver."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.community.extenders.shared.data_access_identity import (
    resolve_data_access_identity,
    sanitize_data_access_identity,
)


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        pytest.param("s3://bucket/key.parquet", "s3://bucket/key.parquet", id="clean_uri_unchanged"),
        pytest.param("/data/dir?/file#1.csv", "/data/dir?/file#1.csv", id="non_uri_unchanged"),
        pytest.param("{host, port}", "{host, port}", id="core_dict_form_unchanged"),
        pytest.param("file:///tmp/data.csv", "file:///tmp/data.csv", id="file_uri_unchanged"),
        pytest.param("https://host/p?sig=SECRET", "https://host/p", id="query"),
        pytest.param("https://host/p#frag", "https://host/p", id="fragment"),
        pytest.param("postgresql://user:pw@db:5432/mydb", "postgresql://db:5432/mydb", id="userinfo"),
        pytest.param(
            "abfss://container@account.dfs.core.windows.net/p",
            "abfss://account.dfs.core.windows.net/p",
            id="azure_container_dropped_with_userinfo",
        ),
        pytest.param("https://u:p@ss@host/db", "https://host/db", id="at_sign_inside_userinfo"),
        pytest.param(
            "jdbc:hive2://h:10000/default;user=u;password=SECRET", "jdbc:hive2://h:10000/default", id="semicolon_params"
        ),
        pytest.param("https://b.com/x&sig=SECRET", "https://b.com/x", id="ampersand_tail_in_path"),
        pytest.param("https://b.com&token=SECRET", "https://b.com", id="authority_leak"),
        pytest.param("jdbc:postgresql://h/db?password=SECRET", "jdbc:postgresql://h/db", id="compound_scheme"),
        pytest.param(
            "postgresql://user:pa?ss@host:5432/db", "postgresql://host:5432/db", id="query_marker_in_password"
        ),
        pytest.param("sftp://user:p@ss?word@host/f", "sftp://host/f", id="at_and_query_marker_in_password"),
        pytest.param("postgresql://user:pa#ss@host/db", "postgresql://host/db", id="fragment_marker_in_password"),
        pytest.param("https://host?x=1@evil/y", "https://evil/y", id="query_with_at_sign"),
    ],
)
def test_sanitize_data_access_identity(identity: str, expected: str) -> None:
    assert sanitize_data_access_identity(identity) == expected


@pytest.mark.parametrize(
    ("args", "context_identity", "expected"),
    [
        pytest.param(
            ("https://host/p?email=a@b.com/x&sig=SECRET",),
            "str",
            "https://host/p",
            id="valid_scheme_raw_wins_over_core_type_name",
        ),
        pytest.param(
            ("postgresql://host/db?user=u&password=p@ss/word",),
            "str",
            "postgresql://host/db",
            id="valid_scheme_raw_password_with_at_sign",
        ),
        pytest.param(("/data/dir?/file#1.csv",), "str", "/data/dir?/file#1.csv", id="non_uri_raw_arg_kept_as_given"),
        pytest.param((), "https://host/p?sig=SECRET", "https://host/p", id="no_args_uses_context"),
        pytest.param((42,), "https://host/p?sig=SECRET", "https://host/p", id="non_str_arg_uses_context"),
        pytest.param(({"k": "v"},), "https://host/db", "https://host/db", id="dict_arg_uses_context"),
        pytest.param((), None, None, id="no_args_and_no_context_is_none"),
        pytest.param((42,), None, None, id="non_str_arg_and_no_context_is_none"),
        pytest.param(
            (" postgresql://user:pw@host/db",),
            "str",
            "str",
            id="whitespace_prefixed_uri_userinfo_falls_back_to_context",
        ),
        pytest.param(
            ("1s3://user:pw@host/db",),
            "str",
            "str",
            id="digit_scheme_uri_userinfo_falls_back_to_context",
        ),
        pytest.param(("/data/dir/file.csv",), "/data/dir/file.csv", "/data/dir/file.csv", id="plain_path_unchanged"),
        pytest.param(
            ("jdbc:postgresql://h/db",),
            "jdbc:postgresql://h",
            "jdbc:postgresql://h/db",
            id="clean_uri_with_differing_core_projection_uses_sanitized_raw",
        ),
        pytest.param(
            ("s3://bucket/my file.parquet",),
            "str",
            "s3://bucket/my file.parquet",
            id="clean_uri_with_space_and_core_type_name_uses_sanitized_raw",
        ),
        pytest.param(
            ("Server=db;Password=SECRET",),
            "str",
            "Server=db;Password=SECRET",
            id="keyword_dsn_raw_arg_kept_as_given",
        ),
    ],
)
def test_resolve_data_access_identity(
    args: tuple[Any, ...], context_identity: str | None, expected: str | None
) -> None:
    assert resolve_data_access_identity(args, context_identity) == expected
