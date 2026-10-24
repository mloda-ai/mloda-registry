"""SourcesFeatureGroup: the collect-of-origins aggregation."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, localcontext
from typing import Any

import pyarrow as pa
from mloda.provider import FeatureChainParserMixin, FeatureGroup, FeatureSet, HookContext, get_all_subclasses
from mloda.user import Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from .reader import ADMISSIBILITY_COLUMN, _ReaderJournal, describe_verdicts, is_admitted, verdict_kind
from .skr import PARTS, SkrAccountFeatureGroup


class InadmissibleTotal(Exception):
    """Rows were asked to produce a total without any policy attesting their admissibility.

    `verdicts` counts the rows by the stamp they carry (or `unstamped`), so the refusal says
    how much of the journal stood in the way.
    """

    def __init__(self, message: str, verdicts: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.verdicts = verdicts


class UncitedValue(Exception):
    """A producer offered an amount with no citation naming the line it came from."""


class AmbiguousSourceColumns(Exception):
    """More than one column could be this concept's amount or its citations."""


def _is_citation(origin: object) -> bool:
    """A citation names a line, and a blank string names nothing."""
    return isinstance(origin, str) and origin.strip() != ""


class SourcesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """``<feature>__sources`` -> the total AND the attested rows behind it.

    Totals are where provenance normally dies: sum reduces rows, and mloda's
    built-in aggregations do not carry row-level origins. Outputs are subcolumns of the
    *requested* name (``revenue__sources~value`` / ``~origins`` / ``~basis``), which is what
    column selection keeps. The input resolves through SkrAccountFeatureGroup unless the options point at a reader.
    A single part such as ``revenue__sources~value`` can be requested; the whole total and its
    refusals still run.
    """

    OUTPUT_PARTS = PARTS
    PREFIX_PATTERN = r".*__sources$"
    RECOGNITION_ONLY_PATTERN = True

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PyArrowTable}

    def input_features(self, options: Options, feature_name: Any) -> set[Feature] | None:
        source = str(feature_name).rsplit("__", 1)[0]
        pointers = {k for reader in get_all_subclasses(_ReaderJournal) for k in reader.pointer_keys()}
        if any(k in options for k in pointers):
            return {Feature(source)}
        return {Feature(source, feature_group=SkrAccountFeatureGroup)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        available = set(data.column_names)

        # No number without a source -- and no number at all without established admissibility.
        # This is the join that makes the four pieces one chain: the aggregation will not total
        # rows whose admissibility no policy has attested. Remove the policy and the pipeline
        # returns nothing, rather than an unvetted total that still looks like money.
        # PRESENCE of the column is the contract, not a non-empty one: a dossier with no rows
        # is still an attested dossier, and must reach the promised null total rather than
        # dying here. Then every stamp must be an AFFIRMATIVE verdict -- checking only for
        # non-null admitted "", "DENIED" and False alike.
        if ADMISSIBILITY_COLUMN not in available:
            raise InadmissibleTotal(
                f"refusing to total: no admissibility evidence in {ADMISSIBILITY_COLUMN!r}. "
                "An admissibility policy (see AdmissibilityPolicyGroup) must stamp the rows "
                "before a citable total is produced.",
                verdicts={"unstamped": data.num_rows},
            )
        stamps = data.column(ADMISSIBILITY_COLUMN).to_pylist()
        bad = [e for e in stamps if not is_admitted(e)]
        if bad:
            counts = dict(Counter(verdict_kind(e) for e in stamps))
            raise InadmissibleTotal(
                f"refusing to total: {len(bad)} row(s) carry no affirmative admissibility "
                f"verdict (e.g. {bad[0]!r}). A blank, negative or malformed stamp is not an "
                f"admission ({describe_verdicts(counts)}).",
                verdicts=counts,
            )

        out: dict[str, pa.Array] = {}
        published: dict[str, str] = {}

        for name in sorted(features.get_all_names()):
            requested = str(name)
            source = requested.rsplit("__", 1)[0]
            columns = cls.resolve_multi_column_feature(source, available)

            # EXACT names, not the first column that happens to end in ~value. A
            # sibling subcolumn such as `revenue~other~value` must never become the total. A
            # plausible number under a valid-looking origin is the precise failure this
            # aggregation exists to prevent, so ambiguity is refused rather than resolved.
            value_col, origin_col = f"{source}~value", f"{source}~origins"
            if value_col not in available or origin_col not in available:
                raise ValueError(
                    f"{requested}: {source} must supply {value_col!r} and {origin_col!r}; "
                    f"got {columns}. A bare source column would drop the citation."
                )
            siblings = sorted(
                c
                for c in columns
                if c not in (value_col, origin_col) and (c.endswith("~value") or c.endswith("~origins"))
            )
            if siblings:
                raise AmbiguousSourceColumns(
                    f"{requested}: {source} offers more than one candidate amount or "
                    f"citation column ({', '.join(siblings)} beside {value_col} and "
                    f"{origin_col}). Refusing rather than picking one: the wrong pick is a "
                    "plausible total under a citation that looks right."
                )

            # Binary floats have no place in a total that claims to be auditable, and
            # Decimal(0) + float is a TypeError from inside the sum rather than a statement
            # about the input. A producer over a source that declares no scale -- a stock
            # CSV reader infers double -- must declare one itself (a decimal128 with its scale).
            declared = data.schema.field(value_col).type
            if pa.types.is_floating(declared):
                raise ValueError(
                    f"{requested}: {value_col} is {declared}, a binary float. Give the "
                    "aggregation a decimal (or integer) column: the producer, not this "
                    "sum, is what knows the scale money is kept at."
                )

            values = data.column(value_col).to_pylist()
            citations = data.column(origin_col).to_pylist()

            # An amount whose line cannot be named must never enter a total that claims to
            # know its rows. A priced row with a null origin would vanish from its own sum,
            # giving a smaller number still cited to a complete-looking list. Unlike an
            # unpriced line, this is a broken producer contract rather than a dossier nobody
            # could price, so it is refused: nulling would let a producer bug pass as an
            # unpriceable source.
            uncited = [v for v, o in zip(values, citations) if v is not None and not _is_citation(o)]
            if uncited:
                raise UncitedValue(
                    f"{requested}: {len(uncited)} row(s) carry an amount with no citation "
                    f"(e.g. {uncited[0]}). A value whose line cannot be named must not enter "
                    "a total that claims to know the rows behind it."
                )

            # Nulls mark ledger lines that do not belong to this concept.
            # Keyed on the CITATION, not the value: a matching line with an empty amount is
            # still an attested row and keeps its citation while adding nothing, so the origin
            # list is the set of contributing LINES, not the set of addends. What the sum then
            # does with that unknown amount is decided below, and it is NOT what SQL does.
            pairs = [(v, o) for v, o in zip(values, citations) if _is_citation(o)]
            origins = [o for _, o in pairs]
            # No contributing line means the concept is unattested in this dossier, which is not
            # the same statement as "the total is zero". A sourceless 0.00 is indistinguishable
            # from a wrong account catalogue, and that is precisely how a chart mix-up hides.
            # A contributing line whose amount is unknown makes the total UNSTATEABLE, not
            # smaller. Skipping it the way SQL's SUM does would return the known subtotal beside
            # a citation list that includes the row nobody could price -- a partial answer
            # wearing the authority of a complete one. That is the sourceless-zero failure with
            # better manners: treating unknown as zero. So one unknown amount among the
            # contributing lines yields null, and the citations still say which rows were found.
            unpriced = sum(1 for v, _ in pairs if v is None)
            amounts = [v for v, _ in pairs if v is not None]

            # Keep the source column's declared scale; a sum must not silently re-scale money.
            source_type = declared
            scale = source_type.scale if pa.types.is_decimal(source_type) else 0

            # Python's default Decimal context rounds at 28 significant digits, which is
            # narrower than the decimal128(38, s) money being summed: on a wide ledger the
            # cents were silently rounded away. Sum with room for the carry, then let Arrow
            # refuse a total that genuinely will not fit the declared type -- loudly, rather
            # than handing back a rounded one.
            precision = source_type.precision if pa.types.is_decimal(source_type) else 38
            if amounts and not unpriced:
                with localcontext() as ctx:
                    ctx.prec = precision + 12
                    total = sum(amounts, Decimal(0))
            else:
                total = None
            out[f"{requested}~value"] = pa.array([total], type=pa.decimal128(38, scale))
            # A list column, not a joined string: an origin is a value, not text to re-split.
            # Null value with a null list is one statement; an empty list beside a null total
            # reads as "we looked and found nothing", which claims more than we know.
            out[f"{requested}~origins"] = pa.array([origins if pairs else None], type=pa.list_(pa.string()))
            basis = cls._basis(data, source)
            out[f"{requested}~basis"] = pa.array([basis], type=pa.string())
            published.update(cls._basis_attributes(requested, basis))
        if published:
            HookContext.publish_result_attributes(published)
        return pa.table(out)

    @staticmethod
    def _basis_attributes(requested: str, basis: str | None) -> dict[str, str]:
        """Result attributes read off a JSON-object basis; anything else publishes nothing."""
        try:
            doc = json.loads(basis) if basis is not None else None
        except ValueError:
            return {}
        if not isinstance(doc, dict):
            return {}
        catalogue = doc.get("catalogue")
        fields = {
            "profile": json.dumps(doc["profile"], sort_keys=True, separators=(",", ":"))
            if doc.get("profile") is not None
            else None,
            "sign": doc.get("sign"),
            "catalogue": catalogue.get("name") if isinstance(catalogue, dict) else None,
            "catalogue.fingerprint": catalogue.get("fingerprint") if isinstance(catalogue, dict) else None,
        }
        return {f"{requested}.{k}": v for k, v in fields.items() if v is not None}

    @staticmethod
    def _basis(data: Any, source: str) -> str | None:
        """The one concept basis the rows carry, or None when they carry none."""
        basis_col = f"{source}~basis"
        if basis_col not in data.column_names:
            return None
        distinct = {b for b in data.column(basis_col).to_pylist() if b is not None}
        # One total, one definition. Two bases means rows computed under two catalogues were joined.
        if len(distinct) > 1:
            raise ValueError(f"{source}: rows carry {len(distinct)} different concept bases; one total has one")
        return distinct.pop() if distinct else None
