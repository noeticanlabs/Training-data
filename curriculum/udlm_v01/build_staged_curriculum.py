from __future__ import annotations

"""Build the deterministic UDLM v0.1 developmental language curriculum.

The curriculum changes environmental complexity, not model architecture.
Stages are intentionally ordered from surface regularity to natural prose/domain
expansion. Each stage is plain UTF-8 and large enough for the certified learner's
within-source acquisition/holdout split.
"""

from pathlib import Path
import shutil

ROOT = Path("udlm_curriculum")


def write_stage(index: int, name: str, lines: list[str]) -> None:
    p = ROOT / f"{index:02d}_{name}" / f"{index:02d}_{name}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).strip() + "\n"
    p.write_text(text, encoding="utf-8")
    print(f"C{index}: {p} bytes={len(text.encode('utf-8')):,}")


def c0_surface() -> list[str]:
    words = [
        "cat", "dog", "book", "tree", "house", "ball", "door", "chair",
        "table", "water", "light", "hand", "road", "bird", "fish", "sun",
        "moon", "red", "blue", "green", "run", "walk", "jump", "read",
    ]
    lines = ["UDLM C0 — surface regularity: letters, spaces, word boundaries, and sentence closure.", ""]
    for _ in range(8):
        lines.extend(words)
        lines.append("")
        lines.extend([f"the {w}" for w in words[:16]])
        lines.extend([f"a {w}" for w in words[:16]])
        lines.append("")
    return lines


def c1_lexical() -> list[str]:
    subjects = ["cat", "dog", "bird", "child", "teacher", "farmer", "driver", "reader"]
    verbs = ["runs", "walks", "rests", "looks", "waits", "jumps", "reads", "listens"]
    lines = ["UDLM C1 — lexical formation: stable words reused across changing local contexts.", ""]
    for _ in range(5):
        for s in subjects:
            lines.extend([
                f"The {s} {verbs[0]}.",
                f"A {s} {verbs[1]}.",
                f"The {s} {verbs[2]}.",
                f"My {s} {verbs[3]}.",
                f"One {s} {verbs[4]}.",
                f"The {s} {verbs[5]}.",
            ])
        lines.append("")
    return lines


def c2_sentences() -> list[str]:
    agents = ["girl", "boy", "teacher", "mother", "father", "worker", "reader", "driver"]
    verbs = ["reads", "opens", "moves", "holds", "sees", "finds", "carries", "checks"]
    objects = ["book", "door", "box", "key", "ball", "cup", "bag", "map"]
    lines = ["UDLM C2 — sentence formation: simple subject, verb, object, location, and closure patterns.", ""]
    for cycle in range(8):
        for i, a in enumerate(agents):
            v = verbs[(i + cycle) % len(verbs)]
            o = objects[(2 * i + cycle) % len(objects)]
            lines.append(f"The {a} {v} the {o}.")
            lines.append(f"The {o} is on the table.")
            lines.append(f"The {a} is near the door.")
        lines.append("")
    return lines


def c3_grammar() -> list[str]:
    pairs = [
        ("dog", "dogs", "runs", "run"),
        ("cat", "cats", "sleeps", "sleep"),
        ("bird", "birds", "flies", "fly"),
        ("child", "children", "reads", "read"),
        ("teacher", "teachers", "walks", "walk"),
        ("worker", "workers", "checks", "check"),
    ]
    lines = ["UDLM C3 — grammar and structure: number agreement, word order, capitalization, punctuation, and complete sentences.", ""]
    for _ in range(14):
        for sg, pl, vsg, vpl in pairs:
            lines.extend([
                f"The {sg} {vsg}.",
                f"The {pl} {vpl}.",
                f"A {sg} is here.",
                f"Two {pl} are here.",
                f"Is the {sg} here?",
                f"Yes, the {sg} is here.",
            ])
        lines.append("")
    return lines


def c4_relations() -> list[str]:
    people = ["Mara", "Leon", "Anna", "Ben", "Nora", "Owen", "Sara", "Mike"]
    objects = ["key", "book", "box", "map", "cup", "ball", "bag", "note"]
    lines = ["UDLM C4 — relations and state transitions: who has what before and after transfer.", ""]
    for cycle in range(10):
        a = people[cycle % len(people)]
        b = people[(cycle + 1) % len(people)]
        c = people[(cycle + 3) % len(people)]
        obj = objects[cycle % len(objects)]
        lines.extend([
            f"{a} has the {obj}.",
            f"{b} has no {obj}.",
            f"{a} gives the {obj} to {b}.",
            f"{b} now has the {obj}.",
            f"{a} no longer has the {obj}.",
            f"{b} gives the {obj} to {c}.",
            f"{c} now has the {obj}.",
            f"{b} no longer has the {obj}.",
            "",
        ])
    return lines


def c5_context() -> list[str]:
    passages = [
        ["Anna went to the store.", "She bought some milk.", "Anna carried the milk home.", "She put it in the refrigerator."],
        ["Ben found a small key.", "He placed the key in his pocket.", "Ben walked to the old door.", "He used the key to open it."],
        ["Nora picked up a red book.", "She read the first page.", "Nora carried the book to the table.", "She left it beside the lamp."],
        ["Owen filled a cup with water.", "He carried the cup carefully.", "Owen crossed the room.", "He set the cup on the desk."],
    ]
    lines = ["UDLM C5 — contextual continuity: entities, pronouns, objects, and facts across several sentences.", ""]
    for _ in range(12):
        for p in passages:
            lines.extend(p)
            lines.append("")
    return lines


def c6_qa() -> list[str]:
    examples = [
        ("Mia put the book on the table.", "Where is the book?", "The book is on the table."),
        ("Ben lost his hat at school.", "What did Ben lose?", "Ben lost his hat."),
        ("Sara gave the key to Mike.", "Who has the key?", "Mike has the key."),
        ("The dog slept under the chair.", "Where did the dog sleep?", "The dog slept under the chair."),
        ("Anna opened the blue door.", "Who opened the door?", "Anna opened the door."),
        ("Owen carried water home.", "What did Owen carry?", "Owen carried water."),
    ]
    lines = ["UDLM C6 — question, answer, and consequence: explicit linguistic dependencies between statements and answers.", ""]
    for _ in range(18):
        for statement, question, answer in examples:
            lines.extend([statement, question, answer, ""])
    return lines


def c7_repair() -> list[str]:
    repairs = [
        ("The dogs is running.", "The dogs are running."),
        ("Cat the sleeps.", "The cat sleeps."),
        ("The boy read book.", "The boy reads the book."),
        ("the girl walks home", "The girl walks home."),
        ("The bird fly.", "The bird flies."),
        ("Two cat are here.", "Two cats are here."),
        ("The chair drank water.", "The child drank water."),
        ("Book the is table on.", "The book is on the table."),
    ]
    lines = ["UDLM C7 — repair and contrast: an explicitly marked broken message followed by a clear corrected message.", ""]
    for _ in range(16):
        for bad, good in repairs:
            lines.extend([
                f"Broken: {bad}",
                f"Repair: {good}",
                f"Ready: {good}",
                "",
            ])
    return lines


def c8_natural() -> list[str]:
    paragraphs = [
        "A small town library opened early each morning. The librarian checked the lights, unlocked the front door, and placed returned books on a cart. Children arrived after school to read, ask questions, and work together at the long tables.",
        "Rain began before sunrise and continued through the afternoon. Water moved along the curb toward the storm drain. By evening the clouds had thinned, the road was wet, and a cool wind moved through the trees.",
        "A maintenance worker received a report that a hallway light was out. He checked the fixture, verified that power was present, replaced the failed lamp, and confirmed that the light operated normally before closing the work order.",
        "Mara carried a box from the office to the storage room. Leon held the door while she walked through. After the box was placed on the shelf, they checked the label and returned to the office.",
        "Language carries information through ordered patterns. Words combine into sentences, sentences connect across context, and readers use those relationships to understand what happened, who was involved, and what may come next.",
    ]
    lines = ["UDLM C8 — natural prose and domain expansion: ordinary multi-sentence English followed by later technical material.", ""]
    for _ in range(12):
        lines.extend(paragraphs)
        lines.append("")
    # Existing technical sources are appended by the workflow after this generated core.
    return lines


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    stages = [
        (0, "surface", c0_surface()),
        (1, "lexical", c1_lexical()),
        (2, "sentences", c2_sentences()),
        (3, "grammar", c3_grammar()),
        (4, "relations", c4_relations()),
        (5, "context", c5_context()),
        (6, "qa", c6_qa()),
        (7, "repair", c7_repair()),
        (8, "natural", c8_natural()),
    ]
    for index, name, lines in stages:
        write_stage(index, name, lines)


if __name__ == "__main__":
    main()
