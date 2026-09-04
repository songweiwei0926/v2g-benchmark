"""Canonical context schema."""
import pandera.polars as pa
import polars as pl

context_schema = pa.DataFrameSchema({
    "context_id": pa.Column(str, nullable=False),
    "context_name": pa.Column(str, nullable=False),
    "context_type": pa.Column(str, pa.Check.isin(["cell_line", "tissue", "primary_cell", "in_vitro", "organoid"]), nullable=False),
    "ontology_id": pa.Column(str, nullable=True),
    "parent_context": pa.Column(str, nullable=True),
    "supergroup": pa.Column(str, nullable=True),
}, strict=True, coerce=True)


def normalize_context_name(name: str) -> str:
    """Normalize a context name for matching."""
    return name.strip().lower().replace("_", " ").replace("-", " ")


def make_context_id(name: str) -> str:
    """Generate a canonical context ID from a name."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")
