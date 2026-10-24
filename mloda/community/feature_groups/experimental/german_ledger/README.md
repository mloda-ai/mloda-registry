# mloda-community-german-ledger

Totals from German ledger exports that carry the ledger lines behind them, or a refusal
that names the reason.

**Status: experimental**, see [Experimental packages](../../../../../docs/packaging.md#experimental-packages)
for what that guarantees. Install with `pip install "mloda-community[german-ledger]"`.

- **Reads:** GDPdU/GoBD dossiers (`index.xml` + data file) and DATEV-Format EXTF 700
  booking batches.
- **Answers:** `revenue__sources` returns the total **and** the cited rows it was summed
  from (`GL.txt@642915ba8de6:1`). A row no admissibility policy cleared never counts.
- **Refuses by name** instead of guessing: bytes that do not decode (GDPdU: the declared code
  page, cp1252 when none is declared; DATEV: UTF-8 or Windows-1252), a post-cutoff entry into
  a closed period, an old DATEV version, overlapping batches, and so on.

## Quickstart

```python
from datetime import date
from pathlib import Path

from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.experimental import german_ledger
from mloda.community.feature_groups.experimental.german_ledger import AdmissibilityPolicyGroup

# The sample dossier exists only in a source checkout (a pip install has no tests/): set your own.
DOSSIER = Path(german_ledger.__file__).parent / "tests" / "fixtures" / "dossier_a"


class Closing2025(AdmissibilityPolicyGroup):
    """Your host policy: FY2025 closed on 31 Dec, nothing keyed in after 15 Jan 2026."""

    LOCK_DATE = date(2026, 1, 15)
    PERIOD_END = date(2025, 12, 31)


results = mloda.run_all(
    features=["revenue__sources"],
    compute_frameworks=[PyArrowTable],
    data_access_collection=DataAccessCollection(folders={str(DOSSIER)}),
)
for table in results:
    print(table.column("revenue__sources~value")[0], table.column("revenue__sources~origins")[0])
```

Three rules keep it predictable:

1. **Install it.** `PluginLoader.all()` discovers the package through its entry point, and
   importing it also registers its groups. `SkrAccountFeatureGroup` then answers `revenue` and
   `receivables` by name alone, and `SourcesFeatureGroup` answers any `<name>__sources`. Scope a
   request with `feature_group=` only when another group also answers the name (e.g. a file
   column called `revenue`), see
   [Names Claimed by an Installed Package](../../../../../docs/guides/01-use-existing-plugin.md#names-claimed-by-an-installed-package).
   A collection holding both formats (GDPdU and DATEV), or several dossiers of one format, must
   point at one source with a reader option key, e.g. `options={"GdpduReader": <folder>}`;
   unpointed, it is refused and the message names that fix.
2. **Exactly one policy per process.** Your subclass of `AdmissibilityPolicyGroup` is the
   policy. With none, `revenue__sources` does not resolve. With two, resolution refuses.
   Either way there is no number. Need several checks? List them as rules in that one
   policy (see below).
3. **Write `from mloda.community... import X`, never `import mloda.community...`.** The
   second form rebinds the name `mloda` and hides `mloda.user.mloda`.

## The chain

```
gdpdu_journal            GdpduReader / DatevExtfReader                    rows (or legs)
gdpdu_journal__admitted  your AdmissibilityPolicyGroup subclass           rows, cleared + stamped
revenue, receivables     SkrAccountFeatureGroup                           concept at line grain
revenue__sources         SourcesFeatureGroup                              total + citations
```

A single part such as `revenue__sources~value` can be requested and returns only that
column. The total is still computed and checked whole, so every refusal still applies.
`revenue~value` is the concept at line grain, not a total.

A `__sources` total is built only over a concept from `SkrAccountFeatureGroup` or a subclass.
The policy is a step in the resolution graph, so it shows in `RunResult.plan`. Removing it is a
resolution failure, not an unvetted total.

## Policy rules

`LOCK_DATE`/`PERIOD_END` is shorthand for one `LateEntryCutoff`. A policy that needs several
checks lists them in `RULES` instead (setting both is a `TypeError`):

```
class DatevClosing2025(AdmissibilityPolicyGroup):
    RULES = (PeriodBound(date(2025, 1, 1), date(2025, 12, 31)), Festschreibung())
```

| Rule | Admits | Outside scope | Refuses |
|---|---|---|---|
| `LateEntryCutoff(lock_date, period_end)` | booked by `period_end`, keyed by `lock_date` | booked after `period_end` | keyed late, never keyed, unbooked, no clock column |
| `PeriodBound(start, end)` | booked within `[start, end]` | booked outside it | unbooked, no booking column |
| `Festschreibung()` | DATEV Festschreibung 1 | Festschreibung 0 | no flag, no column |

All rules run in the one policy step. If any rule refuses, the refusal names every rule that
refused. Otherwise each row gets **one** stamp: `admitted:all-of;rules=…` only when every rule
admitted it, and `outside-scope:all-of;…;outside=<rule>` as soon as one did not. A total
needs every journal row admitted: a single row that is not, outside-scope included, refuses it
(`InadmissibleTotal`). A single rule stamps under its own name.

**One host, both formats.** The late-entry cutoff needs an `Erfassungsdatum` that DATEV does
not carry, and `Festschreibung` needs a flag that GDPdU does not carry. `ForFormat` scopes a
rule to one format's rows:

```
class MixedClosing2025(AdmissibilityPolicyGroup):
    RULES = (
        ForFormat("gdpdu", LateEntryCutoff(date(2026, 1, 15), date(2025, 12, 31))),
        ForFormat("datev", Festschreibung()),
        PeriodBound(date(2025, 1, 1), date(2025, 12, 31)),
    )
```

A scoped rule does not apply to the other format's rows; it neither admits nor excludes them.
A row that no rule applies to is refused, because nothing vouched for it. A row's format comes
from its citation: a DATEV citation names a leg (`/K`, `/G`), a GDPdU citation never does. In
the stamp, the scoped rule is named `gdpdu.late-entry-cutoff`, `datev.festschreibung`.

## Evidence on HookContext

mloda hands each step's declared attributes to extenders as `HookContext.declared_attributes`.
This package declares:

| Group | Keys | Says |
|---|---|---|
| `AdmissibilityPolicyGroup` (policy step) | `policy.verdict` | the policy's affirmative stamp (the verdict a cited total's rows carry, rule parameters included), not this run's outcome |
| `SkrAccountFeatureGroup` (concept step) | `chart`, `catalogue`, `catalogue.fingerprint` | the chart (SKR03/SKR04), the catalogue's name, and a fingerprint over the whole catalogue in force |

The selected profile (which columns were read as account and amount) and the sign convention
depend on the data, so they live in `~basis`, a JSON string on the concept's rows. The total
carries it as `<concept>~basis` beside `~value` and `~origins`, or null when its rows carry
none. Rows computed under two bases are refused: one total has one basis.

Pinning one reader by option key also resolves a single folder that holds both formats.

- **Community route:** `OtelExtender` emits each declared key as a `mloda.declared.<key>` span
  attribute.
- **OpenLineage and audit route:** the declarations reach OpenLineage as the `mlodaDeclared` facet
  on each output dataset (COMPLETE events, so via the community `OpenLineageExtender` and the
  enterprise `LineageFacetsExtender` alike), and the audit record's
  `feature_group_declared_attributes`. A refused policy step emits no COMPLETE event, so it has
  no `mlodaDeclared` facet; its OTel span and audit record (status `error`) still carry the
  declared stamp, which is not the run's outcome.
- **Result attributes:** once all totals of a step are built, each total's profile, sign,
  catalogue and catalogue fingerprint are published as
  `<concept>__sources.profile|sign|catalogue|catalogue.fingerprint` (OTel `mloda.result.<key>`);
  a refused step publishes none, and the `~basis` column stays authoritative.
- **Classification:** both readers declare `mloda.data.classification = confidential`, so an
  audit `ClassificationPolicy` needs clearance `confidential` or above; a gate below it is
  refused. `exempt_readers` skips only the plan-time probe, not the run-start check. Raising the
  clearance is the intended lever. A reader subclass with a lower `CLASSIFICATION` also works:
  enable it in the `PluginCollector` in place of the stock reader, and every dossier it reads
  carries the lower level. A subclass merely defined in the process still takes over every read when no `PluginCollector` excludes it. Where several dossiers of one format must be told apart, only the class-name pointer selects one (no `data_access_handle`).
- **Resolution evidence:** `mloda.diagnose` says, per feature, the group chosen, the groups
  that also matched and those that declined, with stage and reason.

**A refused run is counted too.** `AdmissibilityRefused`, `LateEntryRefused` and
`InadmissibleTotal` carry `verdicts`, the rows counted by kind (`admitted`, `outside-scope`,
`refused`, `unevaluated`, `malformed`, `unstamped`), and say it in the message.

**Mapping to OpenLineage.** The policy step's `mlodaDeclared` output-dataset facet carries
`policy.verdict` and the concept step's carries the catalogue keys. The reader's
`data_access_identity` (e.g. `datev:<folder>@<sha12>`) maps to the input dataset. Nothing here
is tamper-proof: anything inside the process can write these attributes.

## Formats

| | GDPdU / GoBD | DATEV EXTF |
|---|---|---|
| Container | folder with `index.xml` | folder of `EXTF_*.csv` batches, one client |
| Versions | descriptor-driven | Versionsnummer 700, Kategorie 21 (510/300 refused) |
| Rows | one per ledger line | two per booking row: `Konto` (K) and `Gegenkonto` (G) |
| Sign | as declared | +Soll / −Haben on the leg's account; Generalumkehr flips it |
| Citation | `GL.txt@<sha12>:<row>` | `EXTF_….csv@<sha12>:<record>/<K\|G>` |
| Personal data | not read beyond the ledger | Stammdaten (category 16) skipped unread; no Buchungstext |

A folder holding both formats is refused, not resolved to one of them.

## Current limits

- **Concepts:** two stated charts, not a licensed DATEV catalogue. `SKR04_2025`: revenue
  4000–4499, receivables 1200–1249. `SKR03_2025`: revenue 8000–8499, receivables 1400–1449.
  The host names its chart once (`SkrAccountFeatureGroup.CHART = "03"`, default `"04"`). A
  DATEV header that states a different SKR is refused (`ChartConflict`). An empty header
  leaves the host's chart as the only word, so a host must state it: under the wrong chart,
  SKR03 Bank 1200 counts as SKR04 receivables.
- **Gross DATEV revenue:** a revenue leg on an automatic account, or with any BU key other
  than 40, includes VAT. The automatic accounts are those Odoo's `l10n_de` templates give a
  taxable default (SKR03 8196, 8300, 8310, 8315, 8400, 8410; SKR04 4186, 4200, 4300, 4310,
  4315, 4400), kept as facts about the DATEV charts with no Odoo material reproduced. It is
  refused by name (`GrossRevenueRefused`) because no tax-key table exists yet. Only DATEV journals carry a BU
  key; GDPdU lines are booked as declared.
- **Sign:** each concept reports in its natural direction: revenue credit-positive,
  receivables debit-positive. DATEV legs are signed +Soll / −Haben and say so (`Vorzeichen` =
  `soll-positiv`), so revenue legs are flipped. A GDPdU journal states no convention, and its
  amounts are taken as declared, as a Sachkonten export writes them. An exporter that writes
  debit-positive amounts says so on its profile:
  `LedgerProfile("Konto", "Betrag", sign="soll-positiv")`. A profile that contradicts the
  journal's own `Vorzeichen` is refused.
- **Sachkontenlänge:** the charts state 4-digit accounts. A DATEV Sachkonto with more digits
  is refused by name (`AccountLengthUnsupported`) rather than mapped by a guess. No batch we
  hold uses a longer Sachkontenlänge.
- **Twin fixture:** `tests/fixtures/twin_2025`, written by `tests/make_twin.py`, holds the same
  four bookings as a GDPdU dossier and a DATEV batch. Under one `PeriodBound` policy both give
  revenue 22 485,06 and receivables 7 500,00, each with its own citations (`tests/isolated.py:twin`).
- **DATEV totals:** the late-entry cutoff needs an `Erfassungsdatum`, which DATEV does not
  carry, so it refuses every DATEV journal (fail closed). A DATEV host runs `PeriodBound` +
  `Festschreibung` instead.
- **Frameworks:** PyArrow only.

## Development

```bash
pip install -e ".[dev]"
pytest                                   # from the registry root
```

Tests: `test_gdpdu_claims.py` (one claim per test), `test_datev.py`, `test_gdpdu_reader.py` and
`test_policy_rules.py` (the composite policy). The quickstart above runs in the repo doc runner
`tests/test_end2end/test_doc_guide_examples.py`.
Scenarios that register classes process-wide (a shadowing subclass, a second policy,
`PluginLoader.all()`) live in `tests/isolated.py`, each run in a fresh process by its test. Run
one by hand: `python -c "from mloda.community.feature_groups.experimental.german_ledger.tests.isolated import inseparable; print(inseparable())"`.

Other third-party DATEV exports are not committed, as not all are licensed for redistribution.
`test_datev.py` reproduces their format quirks on invented bookings and links each source. The
ledermann sample in `tests/fixtures/` is MIT: its NOTICE carries the license text, and the root
`THIRD_PARTY_NOTICES.md` lists it.
