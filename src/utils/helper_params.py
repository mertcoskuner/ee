"""Declare parameters once and derive their CLI options and validation.

Every parameter dataclass field is created with option(), whose metadata
holds the help text, allowed choices, value bounds, and CLI flag. The CLI
parser (config/args.py) and the dataclass builders read that metadata, so
adding a field to a parameter class is all that is needed to expose it.
"""

import dataclasses


def option(
    default=None,
    help="",
    *,
    choices=None,
    flag=None,
    ge=None,
    gt=None,
    lt=None,
    le=None,
    cli=True,
    allow_all=False,
    default_factory=None,
):
    """Return a dataclass field carrying CLI and validation metadata.

    choices may be a list or a zero-argument callable (for registry names);
    ge / gt / lt / le bound numeric values; flag overrides the CLI name;
    cli=False keeps a derived field out of the parser; allow_all accepts
    "all" in list-valued options.
    """
    metadata = {
        "help": help,
        "choices": choices,
        "flag": flag,
        "ge": ge,
        "gt": gt,
        "lt": lt,
        "le": le,
        "cli": cli,
        "allow_all": allow_all,
    }
    if default_factory is not None:
        return dataclasses.field(default_factory=default_factory, metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def cli_fields(cls):
    """Return the dataclass fields of cls that are exposed on the CLI."""
    return [f for f in dataclasses.fields(cls) if f.metadata.get("cli", True)]


def resolve_choices(field):
    """Return the field's allowed values, calling a choices factory if needed."""
    choices = field.metadata.get("choices")
    return choices() if callable(choices) else choices


def default_of(field):
    """Return the default value of a dataclass field."""
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return field.default


def from_args(cls, args, prefix=""):
    """Build cls from a namespace whose attributes are prefix + field name."""
    return cls(**{f.name: getattr(args, prefix + f.name) for f in cli_fields(cls)})


def bound_errors(field, value, flag):
    """Return messages for every bound in field metadata that value breaks."""
    m, errors = field.metadata, []
    values = value if isinstance(value, (list, tuple)) else [value]
    for v in values:
        if v is None or isinstance(v, (str, bool)):
            continue
        if v != v or v in (float("inf"), float("-inf")):
            errors.append(f"{flag} must be finite")
        if m.get("ge") is not None and not v >= m["ge"]:
            errors.append(f"{flag} must be >= {m['ge']}")
        if m.get("gt") is not None and not v > m["gt"]:
            errors.append(f"{flag} must be > {m['gt']}")
        if m.get("le") is not None and not v <= m["le"]:
            errors.append(f"{flag} must be <= {m['le']}")
        if m.get("lt") is not None and not v < m["lt"]:
            errors.append(f"{flag} must be < {m['lt']}")
    return errors
