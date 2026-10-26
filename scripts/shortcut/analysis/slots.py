"""Rule-based slot extraction for LIBERO instructions (T007).

An instruction becomes a list of actions (verb family, object, destination):

    "pick up the black bowl next to the ramekin and place it on the plate"
        → [Action(verb="place", surface="pick up/place", obj="black bowl", obj_ref="black bowl next to the ramekin",
                  dest=("on", "plate"))]
    "open the top drawer and put the bowl inside"
        → [Action("open", obj="top drawer"), Action("place", obj="black bowl", dest=("in", "top drawer"))]

Clauses split at " and " only before a verb ("between the plate and the ramekin" stays). Pronouns:
"place it" → the previous object, "close it" → the previous destination, "on it" → the previous object,
"put X inside" → into the previously opened object. Names go through CANON (bowl → black bowl, cabinet →
wooden cabinet, cream cheese box → cream cheese, rack → wine rack). Relations: on/on top of → on, in/inside → in,
in front of/to the front of → front, to the right/left of → right/left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VERB_START = r"(?:pick up|place|put|push|open|close|turn on|turn off)\b"
CANON = {
    "bowl": "black bowl",
    "cabinet": "wooden cabinet",
    "cream cheese box": "cream cheese",
    "rack": "wine rack",
    "moka pots": "moka pot",
}
# order matters: longer relations first
RELS = [
    ("on top of", "on"),
    ("in front of", "front"),
    ("to the front of", "front"),
    ("to the right of", "right"),
    ("to the left of", "left"),
    ("on", "on"),
    ("in", "in"),
]
QUALIFIER = re.compile(r"\s+(between|next to|from|in|on)\s+(.*)$")
SIDE = re.compile(r"^(left|right|back|front|top|middle|bottom)\s+(.*)$")


@dataclass(frozen=True)
class Action:
    verb: str  # place | push | open | close | turn on
    surface: str  # verb words as written
    obj: str  # object category ("black bowl")
    obj_ref: str  # object with its locating phrase ("black bowl next to the ramekin")
    dest: tuple[str, str] | None  # (relation, landmark) or None

    @property
    def dest_str(self) -> str:
        return f"{self.dest[0]}:{self.dest[1]}" if self.dest else ""


def normalize(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_det(np_: str) -> str:
    return re.sub(r"^(the|both|a|an)\s+", "", np_.strip())


def canon(name: str) -> str:
    name = strip_det(name)
    name = re.sub(r"^(the)\s+", "", name)
    return CANON.get(name, name)


def landmark(np_: str) -> str:
    """'the top drawer of the wooden cabinet' → 'top drawer'; 'the left plate' → 'plate'; 'top of the cabinet' handled by RELS."""
    np_ = strip_det(np_)
    np_ = (
        re.sub(r"\s+of\s+the\s+.*$", "", np_)
        if re.match(r"^(top|middle|bottom|back)\s+(drawer|compartment)", np_)
        else np_
    )
    m = SIDE.match(np_)
    if m and m.group(2) in ("plate",):
        np_ = m.group(2)
    return canon(np_)


def split_object(np_: str) -> tuple[str, str]:
    """'the black bowl next to the ramekin' → ('black bowl', 'black bowl next to the ramekin')."""
    np_ = strip_det(np_)
    m = QUALIFIER.search(np_)
    if not m:
        c = canon(np_)
        return c, c
    head = canon(np_[: m.start()])
    qual = re.sub(r"\bthe\s+", "", f"{m.group(1)} {m.group(2)}")
    return head, f"{head} {qual}"


def split_dest(rest: str) -> tuple[str, tuple[str, str] | None]:
    """'the bowl on top of the cabinet' → ('the bowl', ('on', 'wooden cabinet')). First relation word wins."""
    best = None
    for words, rel in RELS:
        m = re.search(rf"\s{words}\s", f" {rest} ")
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.end(), rel)
    if best is None:
        return rest, None
    s, e, rel = best
    padded = f" {rest} "
    return padded[:s].strip(), (rel, landmark(padded[e:].strip()))


def parse(text: str) -> list[Action]:
    t = normalize(text)
    clauses = re.split(rf"\s+and\s+(?={VERB_START})", t)
    acts: list[Action] = []
    pending_pick: tuple[str, str] | None = None
    last_obj: tuple[str, str] | None = None
    last_container: str | None = None
    for c in clauses:
        if m := re.match(r"^pick up (.*)$", c):
            pending_pick = split_object(m.group(1))
            last_obj = pending_pick
            continue
        if m := re.match(r"^place it (.*)$", c):
            assert pending_pick, f"place without pick: {text!r}"
            _, dest = split_dest("x " + m.group(1))
            acts.append(Action("place", "pick up/place", pending_pick[0], pending_pick[1], dest))
            pending_pick = None
            continue
        if m := re.match(r"^(open|close|turn on|turn off) (.*)$", c):
            verb, np_ = m.group(1), m.group(2)
            # "close it" → what was just put into (drawer, microwave)
            fallback = last_obj[0] if last_obj else "it"
            name = (last_container or fallback) if np_ == "it" else landmark(np_)
            acts.append(Action(verb, verb, name, name, None))
            last_container = name
            last_obj = (name, name)
            continue
        if m := re.match(r"^(put|push) (.*)$", c):
            verb, rest = m.group(1), m.group(2)
            if rest.endswith(" inside"):
                objs, dest = rest[: -len(" inside")], ("in", last_container or "")
            else:
                objs, dest = split_dest(rest)
                if dest and dest[1] == "it" and last_obj:
                    dest = (dest[0], last_obj[0])
            for o in re.split(r"\s+and\s+", strip_det(objs)) if objs.startswith("both") else [objs]:
                head, ref = split_object(o) if verb == "put" else (canon(o), canon(o))
                acts.append(Action("place" if verb == "put" else "push", verb, head, ref, dest))
                last_obj = (head, ref)
            if dest:
                last_container = dest[1]
            continue
        raise ValueError(f"unparsed clause {c!r} in {text!r}")
    assert pending_pick is None, f"pick without place: {text!r}"
    return acts


def mentions(text: str) -> set[str]:
    """Every object/landmark name in the sentence, any role (used for 'word seen anywhere')."""
    out = set()
    for a in parse(text):
        out.add(a.obj)
        if a.dest:
            out.add(a.dest[1])
        qual = re.sub(r"^(between|next to|from|in|on)\s+", "", a.obj_ref[len(a.obj) :].strip())
        out.update(landmark(q) for q in qual.split(" and ") if q)
    return out
