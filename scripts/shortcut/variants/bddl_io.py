"""Read, edit and write LIBERO BDDL problem files without importing LIBERO.

A BDDL file is one s-expression. `parse` turns it into nested lists of strings, `dump` writes it back in
LIBERO's layout; `Problem` wraps the tree with the edits a variant needs (language, goal, objects, regions,
:init). The LIBERO parser (`robosuite_parse_problem`) reads the output like the original files
(tests/test_variants.py checks the round trip on every Goal / Object task).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Tree = list[Any]  # nested lists of str

# predicate name in BDDL / YAML → arity. LIBERO's parser lower-cases predicates; files use these spellings.
PREDICATES = {"On": 2, "In": 2, "Up": 1, "Open": 1, "Close": 1, "TurnOn": 1, "TurnOff": 1}
_CANON = {k.lower(): k for k in PREDICATES}


def canon_predicate(name: str) -> str:
    key = name.lower()
    if key not in _CANON:
        raise ValueError(f"unknown predicate {name!r} (known: {', '.join(PREDICATES)})")
    return _CANON[key]


def parse_predicate(p: str | list[str] | tuple[str, ...]) -> list[str]:
    """'On a b' / '(On a b)' / ['On', 'a', 'b'] → ['On', 'a', 'b'] with arity checked."""
    toks = p.strip().strip("()").split() if isinstance(p, str) else [str(x) for x in p]
    if not toks:
        raise ValueError("empty predicate")
    name = canon_predicate(toks[0])
    if len(toks) - 1 != PREDICATES[name]:
        raise ValueError(f"{name} takes {PREDICATES[name]} argument(s), got {toks[1:]}")
    return [name, *toks[1:]]


def predicate_str(p: list[str]) -> str:
    return "(" + " ".join(p) + ")"


# ---------------------------------------------------------------- s-expressions


def _tokens(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text)  # comments
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse(text: str) -> Tree:
    toks = _tokens(text)
    stack: list[Tree] = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            if len(stack) < 2:
                raise ValueError("unbalanced ')' in BDDL")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    if len(stack) != 1 or len(stack[0]) != 1:
        raise ValueError("unbalanced '(' in BDDL")
    return stack[0][0]


def _num(x: float) -> str:
    return repr(float(x))


def _flat(node: Any) -> str:
    if isinstance(node, list):
        return "(" + " ".join(_flat(n) for n in node) + ")"
    return str(node)


def dump(tree: Tree) -> str:
    """LIBERO-style layout (sections on their own lines, regions and predicates indented)."""
    head, rest = tree[:2], tree[2:]
    out = [f"(define {_flat(head[1])}"]
    for sec in rest:
        key = sec[0]
        if key in (":domain", ":language"):
            out.append(f"  ({' '.join(map(str, sec))})")
        elif key == ":regions":
            out.append("    (:regions")
            for reg in sec[1:]:
                out.append(f"      ({reg[0]}")
                for prop in reg[1:]:
                    if prop[0] in (":ranges", ":yaw_rotation"):
                        out.append(f"          ({prop[0]} (")
                        for r in prop[1]:
                            out.append("              (" + " ".join(str(v) for v in r) + ")")
                        out.append("            )")
                        out.append("          )")
                    else:
                        out.append(f"          {_flat(prop)}")
                out.append("      )")
            out.append("    )")
        elif key in (":fixtures", ":objects", ":obj_of_interest"):
            out.append("")
            out.append(f"  ({key}")
            items = sec[1:]
            if key == ":obj_of_interest":
                out += [f"    {x}" for x in items]
            else:
                for i in range(0, len(items), 3):
                    out.append("    " + " ".join(items[i : i + 3]))
            out.append("  )")
        elif key == ":init":
            out.append("")
            out.append("  (:init")
            out += [f"    {_flat(p)}" for p in sec[1:]]
            out.append("  )")
        elif key == ":goal":
            out.append("")
            out.append("  (:goal")
            out += [f"    {_flat(p)}" for p in sec[1:]]
            out.append("  )")
        else:
            out.append("")
            out.append(f"  {_flat(sec)}")
    out += ["", ")"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- problem


class Problem:
    """A BDDL problem tree with the edits used by the variant generator."""

    def __init__(self, tree: Tree):
        if not (isinstance(tree, list) and tree and tree[0] == "define"):
            raise ValueError("not a BDDL problem (define ...)")
        self.tree = tree

    @classmethod
    def read(cls, path: str | Path) -> Problem:
        return cls(parse(Path(path).read_text()))

    def copy(self) -> Problem:
        return Problem(copy.deepcopy(self.tree))

    def text(self) -> str:
        return dump(self.tree)

    # -- sections
    def section(self, key: str, create: bool = False) -> Tree:
        for sec in self.tree[2:]:
            if isinstance(sec, list) and sec and sec[0] == key:
                return sec
        if not create:
            raise KeyError(key)
        sec = [key]
        self.tree.append(sec)
        return sec

    @property
    def problem_name(self) -> str:
        return str(self.tree[1][1])

    @property
    def language(self) -> str:
        return " ".join(self.section(":language")[1:])

    def set_language(self, text: str) -> None:
        sec = self.section(":language")
        words = text.split()
        if any(c in text for c in "();"):
            raise ValueError(f"instruction may not contain ( ) ; : {text!r}")
        sec[1:] = words

    def _typed(self, key: str) -> dict[str, str]:
        items = self.section(key)[1:]
        return {items[i]: items[i + 2] for i in range(0, len(items), 3)}

    @property
    def objects(self) -> dict[str, str]:
        """{object name: category} of :objects (movable)."""
        return self._typed(":objects")

    @property
    def fixtures(self) -> dict[str, str]:
        return self._typed(":fixtures")

    @property
    def regions(self) -> dict[str, dict[str, Any]]:
        """{full region name (target_name): {short, target, ranges, yaw_rotation}} as LIBERO names them."""
        out = {}
        for reg in self.section(":regions")[1:]:
            props = {p[0]: p[1:] for p in reg[1:]}
            target = props.get(":target", [None])[0]
            ranges = (
                [[float(v) for v in r] for r in props.get(":ranges", [[]])[0]] if ":ranges" in props else []
            )
            yaw = (
                [[float(v) for v in r] for r in props[":yaw_rotation"][0]] if ":yaw_rotation" in props else []
            )
            out[f"{target}_{reg[0]}"] = {
                "short": reg[0],
                "target": target,
                "ranges": ranges,
                "yaw_rotation": yaw,
            }
        return out

    @property
    def init(self) -> list[list[str]]:
        return [list(p) for p in self.section(":init")[1:]]

    @property
    def goal(self) -> list[list[str]]:
        """Conjunction members of :goal (the files wrap them in (And ...))."""
        sec = self.section(":goal")[1:]
        if len(sec) == 1 and isinstance(sec[0], list) and sec[0] and str(sec[0][0]).lower() == "and":
            return [list(p) for p in sec[0][1:]]
        return [list(p) for p in sec]

    @property
    def obj_of_interest(self) -> list[str]:
        try:
            return list(self.section(":obj_of_interest")[1:])
        except KeyError:
            return []

    def names(self) -> set[str]:
        """Every name a predicate may refer to: objects, fixtures, full region names."""
        return set(self.objects) | set(self.fixtures) | set(self.regions)

    # -- edits
    def set_goal(self, preds: Iterable[list[str]]) -> None:
        sec = self.section(":goal")
        sec[1:] = [["And", *[list(p) for p in preds]]]

    def set_obj_of_interest(self, names: Iterable[str]) -> None:
        sec = self.section(":obj_of_interest", create=True)
        sec[1:] = list(dict.fromkeys(names))

    def remove_object(self, name: str) -> list[str]:
        """Drop an object, its :init predicates, regions that belong to it (e.g. basket_1 contain_region)
        and its obj_of_interest entry. Returns the full names of dropped regions."""
        if name not in self.objects:
            raise ValueError(f"{name} is not an object of this scene (objects: {sorted(self.objects)})")
        objs = self.section(":objects")
        i = objs.index(name)
        del objs[i : i + 3]
        dropped = [full for full, r in self.regions.items() if r["target"] == name]
        regs = self.section(":regions")
        regs[1:] = [r for r in regs[1:] if not any(p[0] == ":target" and p[1] == name for p in r[1:])]
        gone = {name, *dropped}
        init = self.section(":init")
        init[1:] = [p for p in init[1:] if not gone & set(p[1:])]
        if ":obj_of_interest" in [s[0] for s in self.tree[2:] if isinstance(s, list)]:
            self.set_obj_of_interest([x for x in self.obj_of_interest if x != name])
        return dropped

    def add_object(self, name: str, category: str) -> None:
        if name in self.names():
            raise ValueError(f"{name} already exists in this scene")
        self.section(":objects").extend([name, "-", category])

    def add_region(
        self,
        short: str,
        target: str,
        ranges: list[list[float]],
        yaw_rotation: list[list[float]] | None = None,
    ) -> str:
        full = f"{target}_{short}"
        if full in self.regions:
            raise ValueError(f"region {full} already exists")
        reg: Tree = [short, [":target", target], [":ranges", [[_num(v) for v in r] for r in ranges]]]
        if yaw_rotation:
            reg.append([":yaw_rotation", [[_num(v) for v in r] for r in yaw_rotation]])
        self.section(":regions").append(reg)
        return full

    def placement(self, name: str) -> list[str] | None:
        """The (On name X) / (In name X) predicate placing `name` in :init."""
        for p in self.init:
            if len(p) == 3 and p[0].lower() in ("on", "in") and p[1] == name:
                return p
        return None

    def set_placement(self, name: str, region: str, pred: str = "On") -> None:
        init = self.section(":init")
        for i, p in enumerate(init[1:], start=1):
            if len(p) == 3 and str(p[0]).lower() in ("on", "in") and p[1] == name:
                init[i] = [pred, name, region]
                return
        init.append([pred, name, region])

    def set_fixture_state(self, name: str, state: str) -> None:
        """Replace any Open/Close/TurnOn/TurnOff of `name` in :init with `state`."""
        pred = {"open": "Open", "close": "Close", "on": "TurnOn", "off": "TurnOff"}[state]
        init = self.section(":init")
        init[1:] = [
            p
            for p in init[1:]
            if not (
                len(p) == 2 and str(p[0]).lower() in ("open", "close", "turnon", "turnoff") and p[1] == name
            )
        ]
        init.append([pred, name])

    def scene_key(self) -> tuple:
        """What fixes the MuJoCo model and its initial distribution: problem class, fixtures, objects,
        regions and :init. Two problems with the same key can share init states."""
        return (
            self.problem_name,
            sorted(self.fixtures.items()),
            sorted(self.objects.items()),
            sorted((k, repr(v)) for k, v in self.regions.items()),
            sorted(_flat(p) for p in self.init),
        )
