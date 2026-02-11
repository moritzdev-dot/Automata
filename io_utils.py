import os
import re
import tempfile
from typing_extensions import *

from automaton import Automaton, RegularExpression
from grammar import Grammar


def detect_regex(content: str) -> bool:
    lines = [
        line.strip()
        for line in content.strip().split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    # If it starts with regex: or pattern:, it's definitely a regex
    if lines and (
        lines[0].lower().startswith("regex:")
        or lines[0].lower().startswith("pattern:")
    ):
        return True

    # If single line with regex operators and no automaton/grammar keywords
    if len(lines) == 1:
        line = lines[0]
        automaton_keywords = ["type:", "states:", "alphabet:", "start:", "accept:", "->"]
        grammar_keywords = ["::="]

        has_keywords = any(kw in line for kw in automaton_keywords + grammar_keywords)
        has_regex_ops = any(op in line for op in ["*", "+", "(", ")"])

        if has_regex_ops and not has_keywords:
            return True

    return False


def detect_automaton(content: str) -> bool:
    lines = content.strip().split("\n")

    automaton_keywords = ["type:", "states:", "alphabet:", "start:", "accept:"]
    has_automaton_keywords = any(
        any(keyword in line.lower() for keyword in automaton_keywords)
        for line in lines
    )

    if has_automaton_keywords:
        return True

    if "::=" in content:
        return False

    arrow_lines = [line for line in lines if "->" in line or "→" in line]
    if not arrow_lines:
        return False

    automaton_pattern_count = 0
    grammar_pattern_count = 0

    for line in arrow_lines:
        parts = re.split(r"\s*->\s*|\s*→\s*", line)

        if len(parts) == 3:
            automaton_pattern_count += 1
        elif len(parts) == 2:
            lhs = parts[0].strip()
            if len(lhs) == 1 and lhs.isupper():
                grammar_pattern_count += 1
            elif lhs.startswith("q") or lhs.startswith("s") or lhs.startswith("S"):
                automaton_pattern_count += 1

    return automaton_pattern_count > grammar_pattern_count


def load_from_file(
    filename: str,
) -> Tuple[Dict[str, Automaton], Dict[str, Grammar], Dict[str, RegularExpression]]:
    automata: Dict[str, Automaton] = {}
    grammars: Dict[str, Grammar] = {}
    regexes: Dict[str, RegularExpression] = {}

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    name_pattern = re.compile(r"^([A-Za-z]\w*):\s*$", re.MULTILINE)

    if name_pattern.search(content):
        # Named sections: NAME:\n...definition...
        sections = name_pattern.split(content)

        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                continue

            name = sections[i].strip()
            definition = sections[i + 1].strip()

            if not definition:
                continue

            if detect_regex(definition):
                try:
                    pattern = definition.strip()
                    if pattern.lower().startswith("regex:"):
                        pattern = pattern[6:].strip()
                    elif pattern.lower().startswith("pattern:"):
                        pattern = pattern[8:].strip()
                    regexes[name] = RegularExpression(pattern)
                except Exception as e:
                    print(f"Warning: Failed to load regex '{name}': {e}")

            elif detect_automaton(definition):
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", delete=False, suffix=".txt"
                    ) as tmp:
                        tmp.write(definition)
                        tmp_name = tmp.name

                    try:
                        loaded = Automaton.load_from_file(tmp_name)
                        if loaded:
                            automata[name] = loaded[0]
                    finally:
                        os.unlink(tmp_name)
                except Exception as e:
                    print(f"Warning: Failed to load automaton '{name}': {e}")
            else:
                try:
                    grammars[name] = Grammar.from_string(definition)
                except Exception as e:
                    print(f"Warning: Failed to load grammar '{name}': {e}")
    else:
        # Single unnamed item
        base_name = os.path.basename(filename).rsplit(".", 1)[0]

        if detect_regex(content):
            try:
                pattern = content.strip()
                if pattern.lower().startswith("regex:"):
                    pattern = pattern[6:].strip()
                elif pattern.lower().startswith("pattern:"):
                    pattern = pattern[8:].strip()
                regexes[base_name] = RegularExpression(pattern)
            except Exception as e:
                print(f"Warning: Failed to load regex: {e}")

        elif detect_automaton(content):
            try:
                loaded = Automaton.load_from_file(filename)
                if loaded:
                    for idx, aut in enumerate(loaded):
                        key = f"{base_name}{idx if idx > 0 else ''}"
                        automata[key] = aut
            except Exception as e:
                print(f"Warning: Failed to load automaton: {e}")
        else:
            try:
                grammars[base_name] = Grammar.from_file(filename)
            except Exception as e:
                print(f"Warning: Failed to load grammar: {e}")

    return automata, grammars, regexes

