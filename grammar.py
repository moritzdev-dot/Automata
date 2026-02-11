from copy import deepcopy
from enum import Enum
from typing_extensions import *

from automaton import Automaton


class GrammarType(Enum):
    TYPE_0 = 0  # Unrestricted grammar
    TYPE_1 = 1  # Context-sensitive grammar
    TYPE_2 = 2  # Context-free grammar (CFG)
    TYPE_3 = 3  # Regular grammar


class Grammar:
    def __init__(self, grammar_type: GrammarType = GrammarType.TYPE_2):
        self.type: GrammarType = grammar_type
        self.N: Set[str] = set()  # Non-terminals
        self.Sigma: Set[str] = set()  # Terminals
        self.P: Set[Tuple[str, Tuple[str, ...]]] = set()  # Productions
        self.S: str = ""  # Start symbol
        self.EPSILON = "ε"  # Epsilon symbol
    # ------------------------------------------------------------------ #
    # Basic symbol / production management
    # ------------------------------------------------------------------ #

    def add_non_terminal(self, symbol: str):
        self.N.add(symbol)

    def add_terminal(self, symbol: str):
        if symbol == self.EPSILON:
            return
        self.Sigma.add(symbol)

    def set_start_symbol(self, symbol: str):
        self.S = symbol
        if symbol not in self.N:
            self.N.add(symbol)

    def _validate_production(self, lhs: str, rhs: List[str]) -> bool:
        """Validate production rule based on grammar type."""
        if self.type == GrammarType.TYPE_0:
            # Unrestricted: α -> β where α contains at least one non-terminal
            return len(lhs) > 0 and any(symbol in self.N for symbol in lhs)

        if self.type == GrammarType.TYPE_1:
            # Context-sensitive: |α| ≤ |β| (except for S -> ε)
            if len(rhs) == 1 and rhs[0] == self.EPSILON:
                return lhs == self.S
            return len(lhs) <= len(rhs)

        if self.type == GrammarType.TYPE_2:
            # Context-free: A -> α where A is a single non-terminal
            return len(lhs) == 1 and lhs in self.N

        if self.type == GrammarType.TYPE_3:
            # Regular: A -> aB, A -> a, or A -> ε
            if len(lhs) != 1 or lhs not in self.N:
                return False
            if len(rhs) == 1 and rhs[0] == self.EPSILON:
                return True
            if len(rhs) == 1:
                return rhs[0] in self.Sigma
            if len(rhs) == 2:
                return rhs[0] in self.Sigma and rhs[1] in self.N
            return False

        return True

    def add_production(self, lhs: str, rhs: Union[str, List[str]]):
        if isinstance(rhs, str):
            if rhs == self.EPSILON:
                rhs_list = [self.EPSILON]
            else:
                rhs_list = list(rhs)
        else:
            rhs_list = rhs

        if not self._validate_production(lhs, rhs_list):
            raise ValueError(f"Invalid production {lhs} -> {rhs_list} for {self.type.name}")

        self.P.add((lhs, tuple(rhs_list)))

        # Auto-add non-terminals from lhs for Type-2/3
        if self.type in [GrammarType.TYPE_2, GrammarType.TYPE_3] and lhs not in self.N:
            self.N.add(lhs)

    # ------------------------------------------------------------------ #
    # Introspection / pretty-printing
    # ------------------------------------------------------------------ #

    def __str__(self):
        result = f"Grammar Type: {self.type.name}\n"
        result += f"  Non-terminals: {{{', '.join(sorted(self.N))}}}\n"
        result += f"  Terminals: {{{', '.join(sorted(self.Sigma))}}}\n"
        result += f"  Start symbol: {self.S}\n"
        result += "  Productions:\n"

        prod_dict: Dict[str, List[str]] = {}
        for lhs, rhs_tuple in sorted(self.P):
            rhs_str = "".join(rhs_tuple)
            prod_dict.setdefault(lhs, []).append(rhs_str)

        for lhs in sorted(prod_dict.keys()):
            result += f"    {lhs} -> {' | '.join(prod_dict[lhs])}\n"

        return result

    def copy(self) -> "Grammar":
        new_grammar = Grammar(self.type)
        new_grammar.N = deepcopy(self.N)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.P = deepcopy(self.P)
        new_grammar.S = self.S
        new_grammar.EPSILON = self.EPSILON
        return new_grammar

    # ------------------------------------------------------------------ #
    # Type detection / conversion
    # ------------------------------------------------------------------ #

    def detect_grammar_type(self) -> GrammarType:
        is_type3 = True
        is_type2 = True
        is_type1 = True

        for lhs, rhs in self.P:
            rhs_str = "".join(rhs)

            # Type 3 check
            if not (len(lhs) == 1 and lhs in self.N):
                is_type3 = False
            else:
                if rhs != (self.EPSILON,):
                    if len(rhs) >= 1 and rhs[-1] in self.N:
                        u, B = rhs[:-1], rhs[-1]
                        if any(sym not in self.Sigma for sym in u):
                            is_type3 = False
                    else:
                        if any(sym not in self.Sigma for sym in rhs):
                            is_type3 = False

            # Type 2 check
            if not (len(lhs) == 1 and lhs in self.N):
                is_type2 = False

            # Type 1 check
            if rhs != (self.EPSILON,):
                if len(rhs) < len(lhs):
                    is_type1 = False
            else:
                if lhs != self.S:
                    is_type1 = False

        if is_type3:
            return GrammarType.TYPE_3
        if is_type2:
            return GrammarType.TYPE_2
        if is_type1:
            return GrammarType.TYPE_1
        return GrammarType.TYPE_0

    def convert_type(self, target_type: GrammarType) -> "Grammar":
        if self.type == target_type:
            return self.copy()

        # Placeholder: no complex conversions implemented here yet
        if self.type == GrammarType.TYPE_2 and target_type == GrammarType.TYPE_2:
            return self.copy()

        raise NotImplementedError(
            f"Conversion from {self.type.name} to {target_type.name} not implemented"
        )

    # ------------------------------------------------------------------ #
    # Parsing from file / string (simplified format)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_file(
        cls, filename: str, grammar_type: GrammarType = GrammarType.TYPE_2
    ) -> "Grammar":
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_string(content, grammar_type)

    @classmethod
    def from_string(
        cls, content: str, grammar_type: GrammarType = GrammarType.TYPE_2
    ) -> "Grammar":
        return cls._parse_simplified_format(content, grammar_type)

    @classmethod
    def _parse_simplified_format(
        cls, content: str, default_type: GrammarType
    ) -> "Grammar":
        lines = content.strip().split("\n")

        grammar_type = default_type
        specified_start = None
        productions: List[str] = []

        # First pass: collect configuration and productions
        for line in lines:
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()

            if not line:
                continue

            if line.startswith("TYPE:"):
                type_num = int(line.split(":")[1].strip())
                grammar_type = GrammarType(type_num)
                continue

            if line.startswith("START:"):
                specified_start = line.split(":")[1].strip()
                continue

            if line in [
                "PRODUCTIONS:",
                "TERMINALS:",
                "NON_TERMINALS:",
                "NONTERMINALS:",
            ]:
                continue

            if any(
                line.startswith(prefix)
                for prefix in (
                    "TERMINALS:",
                    "NON_TERMINALS:",
                    "NONTERMINALS:",
                    "EPSILON:",
                )
            ):
                continue

            if "->" in line or "→" in line or "::=" in line:
                line = line.replace("→", "->").replace("::=", "->")
                productions.append(line)

        g = cls(grammar_type)

        # Second pass: auto-detect symbols
        all_lhs: Set[str] = set()
        all_symbols: Set[str] = set()

        for prod_line in productions:
            if "->" not in prod_line:
                continue
            parts = prod_line.split("->")
            lhs = parts[0].strip()
            rhs_alternatives = parts[1].strip()

            if lhs.startswith("<") and lhs.endswith(">"):
                lhs = lhs[1:-1]

            all_lhs.add(lhs)

            for rhs in rhs_alternatives.split("|"):
                rhs = rhs.strip()

                i = 0
                while i < len(rhs):
                    if rhs[i] == "<":
                        end = rhs.index(">", i)
                        symbol = rhs[i + 1 : end]
                        all_symbols.add(symbol)
                        i = end + 1
                    elif rhs[i].isspace():
                        i += 1
                    else:
                        all_symbols.add(rhs[i])
                        i += 1

        non_terminals: Set[str] = set()
        for symbol in all_symbols | all_lhs:
            if len(symbol) == 1 and symbol.isupper():
                non_terminals.add(symbol)

        terminals: Set[str] = set()
        epsilon_variants = {"ε", "epsilon", "EPSILON", "λ", ""}
        for symbol in all_symbols:
            if symbol not in non_terminals and symbol not in epsilon_variants:
                terminals.add(symbol)

        for nt in non_terminals:
            g.add_non_terminal(nt)
        for t in terminals:
            g.add_terminal(t)

        # Determine start symbol
        if specified_start:
            start_symbol = specified_start
        elif "S" in non_terminals:
            start_symbol = "S"
        elif all_lhs:
            start_symbol = sorted(all_lhs)[0]
        else:
            raise ValueError("No productions found or unable to determine start symbol")

        g.set_start_symbol(start_symbol)

        # Third pass: add productions
        for prod_line in productions:
            if "->" not in prod_line:
                continue
            parts = prod_line.split("->")
            lhs = parts[0].strip()
            rhs_alternatives = parts[1].strip()

            if lhs.startswith("<") and lhs.endswith(">"):
                lhs = lhs[1:-1]

            for rhs in rhs_alternatives.split("|"):
                rhs = rhs.strip()

                cleaned_rhs = ""
                i = 0
                while i < len(rhs):
                    if rhs[i] == "<":
                        end = rhs.index(">", i)
                        cleaned_rhs += rhs[i + 1 : end]
                        i = end + 1
                    elif rhs[i].isspace():
                        i += 1
                    else:
                        cleaned_rhs += rhs[i]
                        i += 1

                if cleaned_rhs in epsilon_variants:
                    cleaned_rhs = g.EPSILON

                try:
                    g.add_production(lhs, cleaned_rhs)
                except ValueError as e:
                    print(
                        f"Warning: Skipping invalid production '{lhs} -> {cleaned_rhs}': {e}"
                    )

        g.type = g.detect_grammar_type()
        return g

    # ------------------------------------------------------------------ #
    # Type-2 (CFG) specific simplifications and CNF conversion
    # ------------------------------------------------------------------ #

    def remove_non_terminating(self) -> "Grammar":
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_non_terminating only works for Type-2 grammars")

        terminating: Set[str] = set()
        changed = True

        while changed:
            changed = False
            for lhs, rhs_tuple in self.P:
                if lhs in terminating:
                    continue
                all_terminating = True
                for symbol in rhs_tuple:
                    if symbol == self.EPSILON:
                        continue
                    if symbol in self.N and symbol not in terminating:
                        all_terminating = False
                        break
                    if (
                        symbol not in self.N
                        and symbol not in self.Sigma
                        and symbol != self.EPSILON
                    ):
                        all_terminating = False
                        break
                if all_terminating:
                    terminating.add(lhs)
                    changed = True

        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.EPSILON = self.EPSILON

        if self.S not in terminating:
            return new_grammar

        new_grammar.S = self.S

        for nt in terminating:
            new_grammar.N.add(nt)

        for lhs, rhs_tuple in self.P:
            if lhs in terminating:
                if all(
                    not (symbol in self.N and symbol not in terminating)
                    for symbol in rhs_tuple
                ):
                    new_grammar.P.add((lhs, rhs_tuple))

        return new_grammar

    def remove_unreachable(self) -> "Grammar":
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_unreachable only works for Type-2 grammars")

        reachable = {self.S}
        queue = [self.S]

        while queue:
            current = queue.pop(0)
            for lhs, rhs_tuple in self.P:
                if lhs != current:
                    continue
                for symbol in rhs_tuple:
                    if symbol in self.N and symbol not in reachable:
                        reachable.add(symbol)
                        queue.append(symbol)

        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.S = self.S
        new_grammar.EPSILON = self.EPSILON

        for nt in reachable:
            new_grammar.N.add(nt)

        for lhs, rhs_tuple in self.P:
            if lhs not in reachable:
                continue
            if any(symbol in self.N and symbol not in reachable for symbol in rhs_tuple):
                continue
            new_grammar.P.add((lhs, rhs_tuple))

        return new_grammar

    def remove_epsilon_productions(self) -> "Grammar":
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_epsilon_productions only works for Type-2 grammars")

        nullable: Set[str] = set()
        changed = True

        while changed:
            changed = False
            for lhs, rhs_tuple in self.P:
                if lhs in nullable:
                    continue
                if len(rhs_tuple) == 1 and rhs_tuple[0] == self.EPSILON:
                    nullable.add(lhs)
                    changed = True
                else:
                    if rhs_tuple and all(symbol in nullable for symbol in rhs_tuple):
                        nullable.add(lhs)
                        changed = True

        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.EPSILON = self.EPSILON

        start_is_nullable = self.S in nullable

        if start_is_nullable:
            new_start = "S0"
            while new_start in self.N:
                if new_start[-1].isdigit():
                    num = int(new_start[1:]) + 1
                    new_start = f"S{num}"
                else:
                    new_start = new_start + "0"

            new_grammar.S = new_start
            new_grammar.N.add(new_start)
            new_grammar.P.add((new_start, (self.S,)))
            new_grammar.P.add((new_start, (self.EPSILON,)))
        else:
            new_grammar.S = self.S

        for nt in self.N:
            new_grammar.N.add(nt)

        for lhs, rhs_tuple in self.P:
            if len(rhs_tuple) == 1 and rhs_tuple[0] == self.EPSILON:
                continue

            nullable_positions = [
                i for i, symbol in enumerate(rhs_tuple) if symbol in nullable
            ]

            num_nullable = len(nullable_positions)
            for mask in range(1 << num_nullable):
                new_rhs: List[str] = []
                for i, symbol in enumerate(rhs_tuple):
                    if i in nullable_positions:
                        pos_idx = nullable_positions.index(i)
                        if mask & (1 << pos_idx):
                            continue
                    new_rhs.append(symbol)
                if new_rhs:
                    new_grammar.P.add((lhs, tuple(new_rhs)))

        return new_grammar

    def remove_unit_productions(self) -> "Grammar":
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_unit_productions only works for Type-2 grammars")

        unit_pairs: Set[Tuple[str, str]] = set()

        for lhs, rhs_tuple in self.P:
            if len(rhs_tuple) == 1 and rhs_tuple[0] in self.N:
                unit_pairs.add((lhs, rhs_tuple[0]))

        for nt in self.N:
            unit_pairs.add((nt, nt))

        changed = True
        while changed:
            changed = False
            new_pairs: Set[Tuple[str, str]] = set()
            for a, b in unit_pairs:
                for b2, c in unit_pairs:
                    if b == b2 and (a, c) not in unit_pairs:
                        new_pairs.add((a, c))
                        changed = True
            unit_pairs.update(new_pairs)

        new_grammar = Grammar(self.type)
        new_grammar.N = deepcopy(self.N)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.S = self.S
        new_grammar.EPSILON = self.EPSILON

        for lhs, rhs_tuple in self.P:
            if len(rhs_tuple) == 1 and rhs_tuple[0] in self.N:
                continue
            new_grammar.P.add((lhs, rhs_tuple))

        for a, b in unit_pairs:
            for lhs, rhs_tuple in self.P:
                if lhs != b:
                    continue
                if len(rhs_tuple) == 1 and rhs_tuple[0] in self.N:
                    continue
                new_grammar.P.add((a, rhs_tuple))

        return new_grammar

    def to_NEA(self) -> Automaton:
        """Convert Type-3 grammar to NEA."""
        if self.type != GrammarType.TYPE_3:
            raise ValueError("to_NEA only works for Type-3 grammars")

        nea = Automaton()
        nea.type = 4
        new_symbol = "F"
        relation: Set[Tuple[str, str, str]] = set()
        for (A, w) in self.P:
            *u, B = w
            if B in self.N:
                relation.add((A, "".join(u), B))
            else:
                relation.add((A, "".join(w), new_symbol))
        nea.accepting_states = frozenset({new_symbol})
        nea.transition_relation = relation
        nea.start_state = self.S
        nea.states = frozenset(self.N.union({new_symbol}))
        nea.alphabet = self.Sigma
        return nea

    def to_chomsky_normal_form(self) -> "Grammar":
        """Convert CFG to Chomsky Normal Form."""
        if self.type != GrammarType.TYPE_2:
            raise ValueError("to_chomsky_normal_form only works for Type-2 grammars")

        g1 = self.remove_non_terminating()
        g2 = g1.remove_unreachable()
        g3 = g2.remove_epsilon_productions()
        g4 = g3.remove_unit_productions()

        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(g4.Sigma)
        new_grammar.S = g4.S
        new_grammar.EPSILON = g4.EPSILON

        new_nt_counter = 0

        terminal_map: Dict[str, str] = {}

        for lhs, rhs_tuple in g4.P:
            if len(rhs_tuple) == 1 and rhs_tuple[0] == self.EPSILON:
                new_grammar.P.add((lhs, rhs_tuple))
                new_grammar.N.add(lhs)
                continue

            if len(rhs_tuple) == 1 and rhs_tuple[0] in g4.Sigma:
                new_grammar.P.add((lhs, rhs_tuple))
                new_grammar.N.add(lhs)
                continue

            if len(rhs_tuple) == 2 and rhs_tuple[0] in g4.N and rhs_tuple[1] in g4.N:
                new_grammar.P.add((lhs, rhs_tuple))
                new_grammar.N.add(lhs)
                continue

            new_grammar.N.add(lhs)

            converted_rhs: List[str] = []
            for symbol in rhs_tuple:
                if symbol in g4.Sigma:
                    if symbol not in terminal_map:
                        new_nt = f"X{symbol}"
                        terminal_map[symbol] = new_nt
                        new_grammar.N.add(new_nt)
                        new_grammar.P.add((new_nt, (symbol,)))
                    converted_rhs.append(terminal_map[symbol])
                else:
                    converted_rhs.append(symbol)

            if len(converted_rhs) > 2:
                current_lhs = lhs
                for i in range(len(converted_rhs) - 2):
                    new_nt_counter += 1
                    new_nt = f"N{new_nt_counter}"
                    new_grammar.N.add(new_nt)
                    new_grammar.P.add((current_lhs, (converted_rhs[i], new_nt)))
                    current_lhs = new_nt

                new_grammar.P.add(
                    (current_lhs, (converted_rhs[-2], converted_rhs[-1]))
                )
            else:
                new_grammar.P.add((lhs, tuple(converted_rhs)))

        return new_grammar

    # ------------------------------------------------------------------ #
    # PDA conversion (CFG -> PDA with empty stack)
    # ------------------------------------------------------------------ #

    def to_PDA(self) -> Automaton:
        """
        Convert a CFG to an equivalent PDA that accepts by empty stack.

        Construction (standard):
          - Single state q
          - Stack alphabet Γ = N ∪ Σ
          - Start stack symbol = S
          - For each production A -> α, add (q, ε, A) -> (q, α)
          - For each terminal a, add (q, a, a) -> (q, ε)
        """
        if self.type != GrammarType.TYPE_2:
            raise ValueError("to_PDA only defined for Type-2 (CFG) grammars")

        q = "q"

        states = frozenset({q})
        start_state = q
        stack_alphabet: Set[str] = set(self.N) | set(self.Sigma)
        start_stack_symbol = self.S

        transitions: Set[Tuple[str, Optional[str], Optional[str], str, str]] = set()

        # Productions: A -> α  =>  (q, ε, A) -> (q, α)
        for lhs, rhs in self.P:
            if lhs not in self.N:
                continue
            if len(rhs) == 1 and rhs[0] == self.EPSILON:
                # A -> ε  => pop A with epsilon and push nothing
                transitions.add((q, None, lhs, q, ""))
            else:
                alpha = "".join(rhs)
                transitions.add((q, None, lhs, q, alpha))

        # Terminal matching: a on input and stack
        for a in self.Sigma:
            transitions.add((q, a, a, q, ""))

        pda = Automaton(
            type=5,
            states=states,
            alphabet=set(self.Sigma),
            stack_alphabet=stack_alphabet,
            start_state=start_state,
            start_stack_symbol=start_stack_symbol,
            transition_relation=transitions,
            accepting_states=frozenset(),  # empty-stack acceptance
        )

        return pda

    # ------------------------------------------------------------------ #
    # CYK algorithm (membership test for CNF CFG)
    # ------------------------------------------------------------------ #

    def cyk(self, word: str, print_table: bool = True) -> bool:
        """
        CYK membership test for a word.

        This method first converts the grammar to Chomsky Normal Form
        internally (using `to_chomsky_normal_form`) and then runs CYK
        on the CNF grammar, so you can call it on any Type-2 CFG.

        Returns True iff word ∈ L(G). Optionally prints the CYK table.
        """
        if self.type != GrammarType.TYPE_2:
            raise ValueError("CYK is only defined for Type-2 (CFG) grammars")

        # Work on a CNF version of the grammar
        g = self.to_chomsky_normal_form()

        n = len(word)
        if n == 0:
            # Empty word: accepted iff start symbol derives epsilon
            return any(
                lhs == g.S and rhs == (g.EPSILON,)
                for (lhs, rhs) in g.P
            )

        # Index productions for quick access: A -> a, A -> B C
        unary_prods: Dict[str, Set[str]] = {}  # a -> {A}
        binary_prods: Dict[Tuple[str, str], Set[str]] = {}  # (B,C) -> {A}

        for lhs, rhs in g.P:
            if len(rhs) == 1 and rhs[0] != g.EPSILON:
                a = rhs[0]
                unary_prods.setdefault(a, set()).add(lhs)
            elif len(rhs) == 2:
                key = (rhs[0], rhs[1])
                binary_prods.setdefault(key, set()).add(lhs)

        # T[i][j] = set of variables deriving word[i:j+1]
        T: List[List[Set[str]]] = [
            [set() for _ in range(n)] for _ in range(n)
        ]

        # Length 1 substrings
        for i, a in enumerate(word):
            for A in unary_prods.get(a, set()):
                T[i][i].add(A)

        # Length >1 substrings
        for length in range(2, n + 1):  # length of span
            for i in range(n - length + 1):
                j = i + length - 1
                cell_vars: Set[str] = set()
                for k in range(i, j):
                    for B in T[i][k]:
                        for C in T[k + 1][j]:
                            for A in binary_prods.get((B, C), set()):
                                cell_vars.add(A)
                T[i][j].update(cell_vars)

        accepted = g.S in T[0][n - 1]

        if print_table:
            # Build a string table representation
            cell_str: List[List[str]] = [["" for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if j < i:
                        cell_str[i][j] = ""
                    else:
                        syms = sorted(T[i][j])
                        cell_str[i][j] = "{" + ",".join(syms) + "}" if syms else "{}"

            # Column widths (consider header indices as well, using 1-based)
            col_widths: List[int] = []
            for j in range(n):
                max_cell_len = max(len(cell_str[i][j]) for i in range(n))
                max_header_len = len(str(j + 1))
                col_widths.append(max(max_cell_len, max_header_len, 3))

            # Row index column width
            row_w = max(3, len(str(n)))

            def fmt(text: str, width: int) -> str:
                return text.center(width)

            print("\nCYK table (upper-triangular; cell [i,j] is span word[i..j]):")
            # Use 1-based indices for the word display
            print("Word indices:", " ".join(f"{i+1}:{ch}" for i, ch in enumerate(word)))
            print()

            # Helper to build a horizontal separator line
            def build_separator() -> str:
                # +---+-------+-------+ ...
                pieces = ["-" * (row_w + 2)]
                for w in col_widths:
                    pieces.append("-" * (w + 2))
                return "+" + "+".join(pieces) + "+"

            # Header row: column indices (1-based) with '|' separators
            header_cells = [fmt("i\\j", row_w)]
            header_cells += [fmt(str(j + 1), col_widths[j]) for j in range(n)]
            print(build_separator())
            print("| " + " | ".join(header_cells) + " |")
            print(build_separator())

            # Rows with 1-based row indices and '|' separators, plus row separators
            for i in range(n):
                row_cells = [fmt(str(i + 1), row_w)]
                for j in range(n):
                    row_cells.append(fmt(cell_str[i][j], col_widths[j]))
                print("| " + " | ".join(row_cells) + " |")
                print(build_separator())

            print(f"\nResult: word '{word}' is "
                  f"{'ACCEPTED' if accepted else 'REJECTED'} by CYK.\n")

        return accepted

