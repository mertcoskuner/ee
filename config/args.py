"""Parse and validate experiment arguments before loading data.

Options are generated from the parameter dataclasses in src/params: each
field becomes --<prefix><name> (or its metadata flag) with the field's
type, default, choices, and help, grouped by parameter class. Bounds in
the field metadata are checked here, followed by cross-field rules.
"""

import argparse
import types
import typing

from models import MODELS
from src.federated.attacks import FL_ATTACKS
from src.params import PARAM_GROUPS
from src.utils.helper_params import (
    bound_errors,
    cli_fields,
    default_of,
    resolve_choices,
)


def argument_spec(field):
    """Return argparse keyword arguments derived from a field's type."""
    kind = field.type
    if typing.get_origin(kind) in (typing.Union, types.UnionType):
        kind = next(a for a in typing.get_args(kind) if a is not type(None))
    if kind is bool:
        return {"action": "store_true"}
    if typing.get_origin(kind) in (list, tuple):
        return {"type": typing.get_args(kind)[0], "nargs": "+"}
    return {"type": kind}


def add_group(parser, cls, prefix):
    """Add one option per CLI field of the parameter class cls."""
    group = parser.add_argument_group(cls.__doc__.split(":")[0].strip())
    for field in cli_fields(cls):
        spec = argument_spec(field)
        dest = prefix + field.name
        flag = prefix + (field.metadata.get("flag") or field.name)
        choices = resolve_choices(field)
        if choices is not None:
            spec["choices"] = list(choices) + (
                ["all"] if field.metadata.get("allow_all") else []
            )
        shown = default_of(field)
        group.add_argument(
            f"--{flag}",
            dest=dest,
            default=shown,
            help=f"{field.metadata.get('help', '')} (default: {shown})",
            **spec,
        )


def build_parser():
    """Return the argument parser covering every parameter group."""
    parser = argparse.ArgumentParser(
        description="Adversarial attacks, backdoor defenses, and Byzantine-robust "
        "federated learning on MNIST"
    )
    for cls, prefix in PARAM_GROUPS.values():
        add_group(parser, cls, prefix)
    return parser


def cross_field_errors(args):
    """Return messages for settings that are invalid only in combination."""
    errors = []
    if args.mode == "gradcam":
        flat = [
            m for m in MODELS.expand(args.model) if not MODELS.meta(m, "convolutional")
        ]
        if flat:
            errors.append(
                f"--mode gradcam requires convolutional models; not: {' '.join(flat)}"
            )
    attacks = FL_ATTACKS.expand(args.fl_attack)
    hostile = [a for a in attacks if not FL_ATTACKS.meta(a, "benign", False)]
    if hostile and round(args.fl_byzantine_ratio * args.fl_clients) == 0:
        errors.append(
            f"--fl_attack {' '.join(hostile)} needs --fl_byzantine_ratio "
            "with at least one Byzantine client"
        )
    return errors


def args_parser(argv=None):
    """Parse optional CLI tokens and return a validated namespace.

    Use process arguments when argv is None. Invalid values or
    incompatible selections terminate through argparse.error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    errors = []
    for cls, prefix in PARAM_GROUPS.values():
        for field in cli_fields(cls):
            flag = "--" + prefix + (field.metadata.get("flag") or field.name)
            errors += bound_errors(field, getattr(args, prefix + field.name), flag)
    errors += cross_field_errors(args)
    if errors:
        parser.error("; ".join(errors))
    return args
