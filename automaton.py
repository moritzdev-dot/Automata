from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing_extensions import *

from graphviz import Digraph


def powerset_noemptyset(s):
    s = list(s)
    return frozenset(
        {
            frozenset(c)
            for r in range(1, len(s) + 1)
            for c in combinations(s, r)
        }
    )


@dataclass
class Automaton:
    """
    Unified automaton class.

    type:
        1 = DEA
        2 = NEA
        3 = NEA + epsilon
        4 = NEA + words
        5 = PDA
    """

    type: int = 1
    states: frozenset = field(default_factory=frozenset)
    alphabet: Set[str] = field(default_factory=set)
    stack_alphabet: Set[str] = field(default_factory=set)  # For PDA (type 5)
    start_state: Any = None
    start_stack_symbol: str = "Z"  # Initial stack symbol for PDA

    # For types 1-4: Set[Tuple[state, symbol, state]]
    # For type 5 (PDA): Set[Tuple[state, input_symbol, stack_symbol, new_state, stack_string]]
    transition_relation: Set[Tuple] = field(default_factory=set)

    accepting_states: frozenset = field(default_factory=frozenset)
    indices: Dict[Any, int] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Construction / loading
    # -------------------------------------------------------------------------

    @staticmethod
    def load_from_file(file_path: str) -> List["Automaton"]:
        with open(file_path, "r") as f:
            content = f.read()

        automaton_blocks = content.split("---")

        automata = []
        for block in automaton_blocks:
            block = block.strip()
            if not block:
                continue
            automaton = Automaton._parse_block(block)
            automata.append(automaton)

        return automata

    @staticmethod
    def _parse_block(block: str) -> "Automaton":
        states = set()
        alphabet = set()
        stack_alphabet = set()
        start_state = None
        start_stack_symbol = "Z"
        accepting_states = set()
        transitions = set()
        automaton_type = 1  # default DEA

        lines = block.strip().split("\n")

        for line in lines:
            line = line.strip()

            if not line or line.startswith("#"):
                continue
            elif line.startswith("type:"):
                t = line[5:].strip()
                if t.isdigit():
                    automaton_type = int(t)
                else:
                    # Support readable type names
                    type_map = {
                        "dfa": 1,
                        "dea": 1,
                        "nfa": 2,
                        "nea": 2,
                        "enfa": 3,
                        "nfa-e": 3,
                        "epsilon": 3,
                        "nfaw": 4,
                        "word": 4,
                        "pda": 5,
                        "pushdown": 5,
                    }
                    automaton_type = type_map.get(t.lower(), 1)

            # Parse alphabet
            elif line.startswith("alphabet:"):
                symbols = line[9:].strip().split()
                alphabet.update(symbols)

            # Parse stack alphabet (for PDA)
            elif line.startswith("stack_alphabet:") or line.startswith("stack:"):
                prefix_len = 15 if line.startswith("stack_alphabet:") else 6
                symbols = line[prefix_len:].strip().split()
                stack_alphabet.update(symbols)

            # Parse start stack symbol (for PDA)
            elif line.startswith("start_stack:"):
                start_stack_symbol = line[12:].strip()

            # Parse states
            elif line.startswith("states:"):
                state_list = line[7:].strip().split()
                states.update(state_list)

            # Parse start state
            elif line.startswith("start:"):
                start_state = line[6:].strip()

            # Parse accepting states
            elif line.startswith("accept:"):
                acc_list = line[7:].strip().split()
                accepting_states.update(acc_list)

            # Parse transitions
            elif "->" in line:
                if automaton_type == 5:  # PDA transition
                    # Format: q0, a, X -> q1, YZ
                    parts = [p.strip() for p in line.split("->")]
                    if len(parts) == 2:
                        left, right = parts
                        left_parts = [p.strip() for p in left.split(",")]
                        if len(left_parts) == 3:
                            src, input_sym, stack_sym = left_parts
                        else:
                            continue

                        right_parts = [p.strip() for p in right.split(",")]
                        if len(right_parts) == 2:
                            tgt, stack_string = right_parts
                        else:
                            continue

                        # Handle epsilon symbols
                        if input_sym.lower() in ["eps", "epsilon", "ε", "e", ""]:
                            input_sym = None
                        if stack_sym.lower() in ["eps", "epsilon", "ε", "e"]:
                            stack_sym = None
                        if stack_string.lower() in ["eps", "epsilon", "ε", "e", ""]:
                            stack_string = ""

                        transitions.add((src, input_sym, stack_sym, tgt, stack_string))
                else:
                    # Standard FA transition: q0 -> a -> q1
                    parts = [p.strip() for p in line.split("->")]
                    if len(parts) == 3:
                        src, symbol, tgt = parts
                    elif len(parts) == 2:
                        src, symbol = parts
                        tgt = None
                    else:
                        continue

                    # Handle epsilon symbol
                    if symbol.lower() in ["eps", "epsilon", "ε", "e", ""]:
                        symbol = None

                    transitions.add((src, symbol, tgt))

            elif len(line.split()) >= 3:
                parts = line.split()

                if automaton_type == 5 and len(parts) >= 5:
                    # PDA format: q0 a X q1 YZ
                    src = parts[0]
                    input_sym = parts[1]
                    stack_sym = parts[2]
                    tgt = parts[3]
                    stack_string = parts[4]

                    # Handle epsilon symbols
                    if input_sym.lower() in ["eps", "epsilon", "ε", "e"]:
                        input_sym = None
                    if stack_sym.lower() in ["eps", "epsilon", "ε", "e"]:
                        stack_sym = None
                    if stack_string.lower() in ["eps", "epsilon", "ε", "e"]:
                        stack_string = ""

                    transitions.add((src, input_sym, stack_sym, tgt, stack_string))
                else:
                    # Standard FA format: q0 a q1
                    src = parts[0]
                    symbol = parts[1]
                    tgt = parts[2]

                    # Handle epsilon symbol
                    if symbol.lower() in ["eps", "epsilon", "ε", "e"]:
                        symbol = None

                    transitions.add((src, symbol, tgt))

        # Build indices
        indices = {s: i for i, s in enumerate(sorted(states))}

        return Automaton(
            type=automaton_type,
            states=frozenset(states),
            alphabet=alphabet,
            stack_alphabet=stack_alphabet,
            start_state=start_state,
            start_stack_symbol=start_stack_symbol,
            accepting_states=frozenset(accepting_states),
            transition_relation=transitions,
            indices=indices,
        )

    def __post_init__(self):
        """Validate the automaton type and structure."""
        if self.type not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Invalid automaton type: {self.type}. Must be 1, 2, 3, 4, or 5."
            )

        if self.type == 5 and not self.stack_alphabet:
            raise ValueError("PDA (type 5) requires a stack_alphabet")

    # -------------------------------------------------------------------------
    # Transition helpers
    # -------------------------------------------------------------------------

    def _get_targets(self, state: Any, symbol: Any) -> frozenset:
        """Get all target states for a given (state, symbol) pair (non-PDA)."""
        if self.type == 5:
            raise ValueError("Use _get_pda_transitions for PDA (type 5)")

        return frozenset(
            {
                target
                for (src, sym, target) in self.transition_relation
                if src == state and sym == symbol
            }
        )

    def _get_pda_transitions(
        self, state: Any, input_symbol: Any, stack_symbol: Any
    ) -> Set[Tuple[Any, str]]:
        """
        Get all possible PDA transitions for a given configuration.
        Returns set of (new_state, stack_string) tuples.
        """
        if self.type != 5:
            raise ValueError("This method is only for PDA (type 5)")

        return {
            (new_state, stack_string)
            for (src, inp, stk, new_state, stack_string) in self.transition_relation
            if src == state and inp == input_symbol and stk == stack_symbol
        }

    def _get_transition_dict(self) -> Dict[Tuple[Any, Any], frozenset]:
        """Convert transition relation to dictionary format (non-PDA)."""
        if self.type == 5:
            raise ValueError("Use _get_pda_transition_dict for PDA (type 5)")

        result = defaultdict(set)
        for src, sym, tgt in self.transition_relation:
            result[(src, sym)].add(tgt)
        return {k: frozenset(v) for k, v in result.items()}

    def _get_pda_transition_dict(
        self,
    ) -> Dict[Tuple[Any, Any, Any], Set[Tuple[Any, str]]]:
        """Convert PDA transition relation to dictionary format."""
        if self.type != 5:
            raise ValueError("This method is only for PDA (type 5)")

        result = defaultdict(set)
        for src, inp, stk, new_state, stack_string in self.transition_relation:
            result[(src, inp, stk)].add((new_state, stack_string))
        return dict(result)

    # -------------------------------------------------------------------------
    # PDA-specific methods
    # -------------------------------------------------------------------------

    def accepts_pda(self, word: str, acceptance_mode: str = "final_state") -> bool:
        """
        Check if PDA accepts a word.
        acceptance_mode: 'final_state' or 'empty_stack'
        """
        if self.type != 5:
            raise ValueError("This method is only for PDA (type 5)")

        trans_dict = self._get_pda_transition_dict()

        # Configuration: (state, remaining_input, stack)
        # Stack is represented as a string (rightmost = top)
        initial_config = (self.start_state, word, self.start_stack_symbol)

        # DFS to explore all possible computation paths
        stack = [initial_config]
        visited = set()

        while stack:
            state, remaining, pda_stack = stack.pop()

            config_key = (state, remaining, pda_stack)
            if config_key in visited:
                continue
            visited.add(config_key)

            # Check acceptance conditions
            if not remaining:  # All input consumed
                if acceptance_mode == "final_state":
                    if state in self.accepting_states:
                        return True
                elif acceptance_mode == "empty_stack":
                    if not pda_stack:
                        return True

            # If stack is empty, only epsilon transitions without stack requirements can proceed
            if not pda_stack:
                for (new_state, stack_string) in trans_dict.get(
                    (state, None, None), set()
                ):
                    new_config = (new_state, remaining, stack_string)
                    stack.append(new_config)
                continue

            # Current stack top
            stack_top = pda_stack[-1]

            # Epsilon transitions (without consuming input)
            for (new_state, stack_string) in trans_dict.get(
                (state, None, stack_top), set()
            ):
                new_stack = pda_stack[:-1] + stack_string
                new_config = (new_state, remaining, new_stack)
                stack.append(new_config)

            # Transitions with current input symbol (if any remaining)
            if remaining:
                current_symbol = remaining[0]
                for (new_state, stack_string) in trans_dict.get(
                    (state, current_symbol, stack_top), set()
                ):
                    new_stack = pda_stack[:-1] + stack_string
                    new_config = (new_state, remaining[1:], new_stack)
                    stack.append(new_config)

        return False

    def accepts_pda_final_state(self, word: str) -> bool:
        """Convenience: PDA acceptance by final state."""
        return self.accepts_pda(word, acceptance_mode="final_state")

    def accepts_pda_empty_stack(self, word: str) -> bool:
        """Convenience: PDA acceptance by empty stack."""
        return self.accepts_pda(word, acceptance_mode="empty_stack")

    def simulate_pda(self, word: str, max_steps: int = 1000) -> List[Tuple[Any, str, str]]:
        """
        Simulate PDA computation and return one possible computation path.
        Returns list of (state, remaining_input, stack) configurations.
        """
        if self.type != 5:
            raise ValueError("This method is only for PDA (type 5)")

        trans_dict = self._get_pda_transition_dict()

        # Configuration: (state, remaining_input, stack, path)
        initial_config = (
            self.start_state,
            word,
            self.start_stack_symbol,
            [(self.start_state, word, self.start_stack_symbol)],
        )

        queue = [initial_config]
        visited = set()
        steps = 0

        while queue and steps < max_steps:
            steps += 1
            state, remaining, pda_stack, path = queue.pop(0)

            config_key = (state, remaining, pda_stack)
            if config_key in visited:
                continue
            visited.add(config_key)

            # Check if we've accepted
            if not remaining and state in self.accepting_states:
                return path

            if not pda_stack:
                for (new_state, stack_string) in trans_dict.get(
                    (state, None, None), set()
                ):
                    new_config = (new_state, remaining, stack_string)
                    new_path = path + [new_config]
                    queue.append((new_state, remaining, stack_string, new_path))
                continue

            stack_top = pda_stack[-1]

            # Epsilon transitions
            for (new_state, stack_string) in trans_dict.get(
                (state, None, stack_top), set()
            ):
                new_stack = pda_stack[:-1] + stack_string
                new_config = (new_state, remaining, new_stack)
                new_path = path + [new_config]
                queue.append((new_state, remaining, new_stack, new_path))

            # Input symbol transitions
            if remaining:
                current_symbol = remaining[0]
                for (new_state, stack_string) in trans_dict.get(
                    (state, current_symbol, stack_top), set()
                ):
                    new_stack = pda_stack[:-1] + stack_string
                    new_config = (new_state, remaining[1:], new_stack)
                    new_path = path + [new_config]
                    queue.append((new_state, remaining[1:], new_stack, new_path))

        # Return start configuration path if no accepting path found
        return [(self.start_state, word, self.start_stack_symbol)]

    # -------------------------------------------------------------------------
    # DEA-specific methods
    # -------------------------------------------------------------------------

    def complement(self) -> "Automaton":
        """Compute complement (only for DEA)."""
        if self.type != 1:
            raise ValueError("Complement operation only defined for DEA (type 1)")

        return Automaton(
            type=1,
            start_state=self.start_state,
            alphabet=self.alphabet,
            transition_relation=self.transition_relation,
            states=self.states,
            accepting_states=self.states.difference(self.accepting_states),
        )

    def intersection(self, other: "Automaton") -> "Automaton":
        """Compute intersection (only for DEA)."""
        if self.type != 1 or other.type != 1:
            raise ValueError("Intersection operation only defined for DEA (type 1)")

        new_states = frozenset(
            {(S, S2) for S in self.states for S2 in other.states}
        )

        new_relation = set()
        for src1, sym1, tgt1 in self.transition_relation:
            for src2, sym2, tgt2 in other.transition_relation:
                if sym1 == sym2:
                    new_relation.add(((src1, src2), sym1, (tgt1, tgt2)))

        return Automaton(
            type=1,
            states=new_states,
            transition_relation=new_relation,
            alphabet=self.alphabet,
            accepting_states=frozenset(
                {(S, S2) for S in self.accepting_states for S2 in other.accepting_states}
            ),
            start_state=(self.start_state, other.start_state),
        )

    def compute_equivalence_classes(self) -> Set[frozenset]:
        """
        Compute equivalence classes by iterative refinement (only for DEA).
        This is the same as finding the largest k where ~_k = ~_(k+1)
        """
        if self.type != 1:
            raise ValueError(
                "Equivalence class computation only defined for DEA (type 1)"
            )

        # Build transition lookup
        trans_dict = self._get_transition_dict()

        # Start with ~_0
        accepting_class = frozenset(self.accepting_states)
        non_accepting_class = self.states - self.accepting_states

        current_classes = set()
        if accepting_class:
            current_classes.add(accepting_class)
        if non_accepting_class:
            current_classes.add(non_accepting_class)

        while True:
            # Map each state to its current equivalence class
            state_to_class = {}
            for eq_class in current_classes:
                for state in eq_class:
                    state_to_class[state] = eq_class

            # Refine classes
            new_classes = set()

            for old_class in current_classes:
                # Group states by their transition signature
                signature_groups = defaultdict(set)

                for state in old_class:
                    # Create signature: where does each symbol take us?
                    signature = []
                    for symbol in sorted(self.alphabet):
                        next_states = trans_dict.get((state, symbol), frozenset())
                        if next_states:
                            # For DEA, should be exactly one state
                            next_state = next(iter(next_states))
                            next_class = state_to_class[next_state]
                            signature.append(id(next_class))
                        else:
                            signature.append(None)

                    signature_groups[tuple(signature)].add(state)

                # Each distinct signature becomes a new class
                for group in signature_groups.values():
                    new_classes.add(frozenset(group))

            # Check for fixed point
            if new_classes == current_classes:
                return current_classes

            current_classes = new_classes

    def minimize(self) -> "Automaton":
        """Minimize DEA."""
        if self.type != 1:
            raise ValueError("Minimization only defined for DEA (type 1)")

        equiv_classes = self.compute_equivalence_classes()

        state_to_class = {}
        for eq_class in equiv_classes:
            for state in eq_class:
                state_to_class[state] = eq_class

        new_states = frozenset(equiv_classes)

        # Build new transitions
        new_relation = set()
        for eq_class in equiv_classes:
            representative = next(iter(eq_class))

            for symbol in self.alphabet:
                next_states = self._get_targets(representative, symbol)
                if next_states:
                    next_state = next(iter(next_states))  # DEA has exactly one
                    next_class = state_to_class[next_state]
                    new_relation.add((eq_class, symbol, next_class))

        new_start = state_to_class[self.start_state]

        new_accepting = frozenset(
            {
                eq_class
                for eq_class in equiv_classes
                if any(state in self.accepting_states for state in eq_class)
            }
        )

        return Automaton(
            type=1,
            states=new_states,
            alphabet=self.alphabet,
            start_state=new_start,
            transition_relation=new_relation,
            accepting_states=new_accepting,
        )

    # -------------------------------------------------------------------------
    # Conversions
    # -------------------------------------------------------------------------

    def to_regex(self) -> "RegularExpression":
        """
        Convert automaton to regular expression using state elimination.
        Only works for finite automata (types 1, 2, 3, 4).
        """
        if self.type == 5:
            raise ValueError("Cannot convert PDA to regular expression")

        if self.type == 4:
            # Convert to type 3 first
            return self._eliminate_word_transitions().to_regex()

        if self.type == 3:
            # Convert to type 2 first
            return self._eliminate_epsilon().to_regex()

        if self.type in (1, 2):
            # Use state elimination algorithm
            return self._state_elimination_to_regex()

        raise ValueError("Cannot convert this automaton type to regex")

    def _state_elimination_to_regex(self) -> "RegularExpression":
        # Create a copy with generalized transitions (edges can be regexes)
        # transitions: dict[(state, state)] = regex_string
        transitions: Dict[Tuple[Any, Any], str] = {}

        # Initialize with existing transitions
        trans_dict = defaultdict(list)
        for src, sym, tgt in self.transition_relation:
            if sym is None:
                trans_dict[(src, tgt)].append("ε")
            else:
                trans_dict[(src, tgt)].append(str(sym))

        # Combine parallel edges with +
        for (src, tgt), symbols in trans_dict.items():
            if len(symbols) == 1:
                transitions[(src, tgt)] = symbols[0]
            else:
                transitions[(src, tgt)] = "(" + "+".join(symbols) + ")"

        # Create new unique start and accept states
        new_start = "_start"
        new_accept = "_accept"

        # Add epsilon transitions from new start to old start
        transitions[(new_start, self.start_state)] = "ε"

        # Add epsilon transitions from old accepting states to new accept
        for acc in self.accepting_states:
            transitions[(acc, new_accept)] = "ε"

        # States to eliminate (all except new start and accept)
        states_to_eliminate = list(self.states)

        # Eliminate states one by one
        for state in states_to_eliminate:
            # Find all transitions going through this state
            incoming = [
                (s, state) for (s, t) in list(transitions.keys()) if t == state and s != state
            ]
            outgoing = [
                (state, t) for (s, t) in list(transitions.keys()) if s == state and t != state
            ]
            self_loop = transitions.get((state, state), None)

            # For each pair of incoming and outgoing edges
            for (src, _) in incoming:
                for (_, tgt) in outgoing:
                    # Build new regex: incoming · (self_loop)* · outgoing
                    parts: List[str] = []

                    in_regex = transitions.get((src, state), "")
                    out_regex = transitions.get((state, tgt), "")

                    if in_regex:
                        parts.append(
                            in_regex
                            if "+" not in in_regex or in_regex.startswith("(")
                            else f"({in_regex})"
                        )

                    if self_loop:
                        loop = (
                            self_loop
                            if "+" not in self_loop or self_loop.startswith("(")
                            else f"({self_loop})"
                        )
                        parts.append(f"({loop})*")

                    if out_regex:
                        parts.append(
                            out_regex
                            if "+" not in out_regex or out_regex.startswith("(")
                            else f"({out_regex})"
                        )

                    new_regex = "".join(parts)

                    # Add or combine with existing transition
                    if (src, tgt) in transitions:
                        existing = transitions[(src, tgt)]
                        transitions[(src, tgt)] = f"({existing}+{new_regex})"
                    else:
                        transitions[(src, tgt)] = new_regex

            # Remove all transitions involving the eliminated state
            keys_to_remove = [
                (s, t)
                for (s, t) in list(transitions.keys())
                if s == state or t == state
            ]
            for key in keys_to_remove:
                del transitions[key]

        # The final regex is the transition from new_start to new_accept
        final_regex = transitions.get((new_start, new_accept), "∅")

        # Simplify epsilon
        final_regex = final_regex.replace("εε", "ε").replace("ε", "")
        if not final_regex:
            final_regex = "ε"

        return RegularExpression(final_regex)

    def to_DEA(self) -> "Automaton":
        """Convert NEA (types 2, 3, 4) to DEA using powerset construction."""
        if self.type == 1:
            return self

        if self.type == 5:
            raise ValueError(
                "Cannot convert PDA to DEA (PDAs recognize context-free languages)"
            )

        if self.type == 3:
            return self._eliminate_epsilon().to_DEA()

        if self.type == 4:
            return self._eliminate_word_transitions().to_DEA()

        new_start = frozenset({self.start_state})
        trans_dict = self._get_transition_dict()

        new_relation = set()
        reachable_states: Set[frozenset] = set()
        queue = [new_start]
        reachable_states.add(new_start)

        while queue:
            S = queue.pop(0)

            for a in self.alphabet:
                target_set = set()
                for q in S:
                    target_set.update(trans_dict.get((q, a), frozenset()))

                if target_set:
                    target_frozenset = frozenset(target_set)
                    new_relation.add((S, a, target_frozenset))

                    if target_frozenset not in reachable_states:
                        reachable_states.add(target_frozenset)
                        queue.append(target_frozenset)

        accepting = frozenset(
            {
                S
                for S in reachable_states
                if len(S.intersection(self.accepting_states)) != 0
            }
        )

        return Automaton(
            type=1,
            states=frozenset(reachable_states),
            accepting_states=accepting,
            start_state=new_start,
            transition_relation=new_relation,
            alphabet=self.alphabet,
        )

    def _eliminate_epsilon(self) -> "Automaton":
        if self.type != 3:
            return self

        trans_dict = self._get_transition_dict()

        def epsilon_closure(state):
            closure = {state}
            stack = [state]
            while stack:
                s = stack.pop()
                for next_state in trans_dict.get((s, None), frozenset()):
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
            return frozenset(closure)

        new_relation = set()
        for state in self.states:
            eps_closure = epsilon_closure(state)
            for symbol in self.alphabet:
                if symbol is None:  # Skip epsilon
                    continue
                targets = set()
                for s in eps_closure:
                    for target in trans_dict.get((s, symbol), frozenset()):
                        targets.update(epsilon_closure(target))
                for target in targets:
                    new_relation.add((state, symbol, target))

        new_accepting = set(self.accepting_states)
        for state in self.states:
            if epsilon_closure(state).intersection(self.accepting_states):
                new_accepting.add(state)

        return Automaton(
            type=2,
            states=self.states,
            alphabet=self.alphabet,
            start_state=self.start_state,
            transition_relation=new_relation,
            accepting_states=frozenset(new_accepting),
        )

    def _eliminate_word_transitions(self) -> "Automaton":
        """Eliminate word transitions (convert type 4 to type 3)."""
        if self.type != 4:
            return self

        new_states = set(self.states)
        new_relation = set()
        state_counter = 0

        for src, word, tgt in self.transition_relation:
            if word is None or len(word) == 0:  # Epsilon
                new_relation.add((src, None, tgt))
            elif len(word) == 1:  # Single symbol
                new_relation.add((src, word, tgt))
            else:
                prev_state = src
                for symbol in word[:-1]:
                    intermediate = f"_q{state_counter}"
                    state_counter += 1
                    new_states.add(intermediate)
                    new_relation.add((prev_state, symbol, intermediate))
                    prev_state = intermediate
                new_relation.add((prev_state, word[-1], tgt))

        return Automaton(
            type=3,
            states=frozenset(new_states),
            alphabet=self.alphabet,
            start_state=self.start_state,
            transition_relation=new_relation,
            accepting_states=self.accepting_states,
        )

    def to_CFG(self) -> "Grammar":
        """
        Convert an arbitrary PDA (type 5) to an equivalent CFG.
        
        This uses the standard construction for PDAs that accept by empty stack.
        If your PDA was obtained from `Grammar.to_PDA`, this will recover an
        equivalent CFG (up to standard transformations).
        
        Construction sketch:
          - Non-terminals: one for each triple (p, A, q) written as X_p_A_q
            meaning: starting in state p with A on top of the stack, we can
            consume some word and end in state q with A popped.
          - Start symbol S with productions S -> X_q0_Z0_q for every state q.
          - For transitions:
              (p, a, A -> r, ε):
                  X_p_A_r -> a        (or ε if a is epsilon)
              (p, a, A -> r, B1...Bk), k >= 1:
                  for all sequences of states q0 = r, q1, ..., qk = q:
                      X_p_A_q -> a X_r_B1_q1 X_q1_B2_q2 ... X_q{k-1}_Bk_q
        """
        if self.type != 5:
            raise ValueError("to_CFG only supported for PDA (type 5)")

        # Local import to avoid circular dependency at module import time
        from grammar import Grammar, GrammarType

        g = Grammar(GrammarType.TYPE_2)

        states = list(self.states)
        stack_symbols = list(self.stack_alphabet)

        def var(p: Any, A: str, q: Any) -> str:
            """Name of non-terminal representing (p, A, q)."""
            return f"X_{p}_{A}_{q}"

        # Create all non-terminals X_p_A_q
        for p in states:
            for q in states:
                for A in stack_symbols:
                    g.add_non_terminal(var(p, A, q))

        # Terminals come from PDA input alphabet
        for a in self.alphabet:
            g.Sigma.add(a)

        # Start symbol S (add manually to avoid strict Type-2 validation)
        g.S = "S"
        g.N.add("S")

        # S -> X_{q0,Z0,q} for all q  (insert directly into P)
        for q in states:
            g.N.add(var(self.start_state, self.start_stack_symbol, q))
            g.P.add(("S", (var(self.start_state, self.start_stack_symbol, q),)))

        # Build productions from PDA transitions
        for (p, inp, stk, r, stack_string) in self.transition_relation:
            # We only handle transitions that pop a concrete stack symbol
            if stk is None or stk not in self.stack_alphabet:
                continue

            a_sym = None if inp is None else str(inp)
            gamma = stack_string or ""
            k = len(gamma)

            # Case 1: pop A and push nothing (k == 0)
            if k == 0:
                # X_p_A_r -> a  or  -> ε if a_sym is None
                lhs = var(p, stk, r)
                g.N.add(lhs)
                if a_sym is None:
                    g.P.add((lhs, (g.EPSILON,)))
                else:
                    g.P.add((lhs, (a_sym,)))
                continue

            # Case 2: push at least one symbol: gamma = B1...Bk
            symbols = list(gamma)

            # For each target state q and each sequence of intermediate states
            # q0 = r, q1, ..., q_{k-1}, q_k = q
            for q in states:
                if k == 1:
                    # X_p_A_q -> a X_r_B1_q
                    B1 = symbols[0]
                    rhs_syms: List[str] = []
                    if a_sym is not None:
                        rhs_syms.append(a_sym)
                    rhs_syms.append(var(r, B1, q))
                    lhs = var(p, stk, q)
                    g.N.add(lhs)
                    g.N.add(var(r, B1, q))
                    g.P.add((lhs, tuple(rhs_syms)))
                else:
                    # k >= 2: iterate over all (q1,...,q_{k-1})
                    for middle in product(states, repeat=k - 1):
                        rhs_syms: List[str] = []
                        if a_sym is not None:
                            rhs_syms.append(a_sym)

                        # First variable: X_r_B1_q1
                        q1 = middle[0]
                        v1 = var(r, symbols[0], q1)
                        g.N.add(v1)
                        rhs_syms.append(v1)

                        # Middle variables
                        for idx in range(1, k - 1):
                            left_state = middle[idx]
                            prev_state = middle[idx - 1]
                            B = symbols[idx]
                            v_mid = var(prev_state, B, left_state)
                            g.N.add(v_mid)
                            rhs_syms.append(v_mid)

                        # Last variable: X_{q_{k-1}}_Bk_q
                        last_mid = middle[-1]
                        v_last = var(last_mid, symbols[-1], q)
                        g.N.add(v_last)
                        rhs_syms.append(v_last)

                        lhs = var(p, stk, q)
                        g.N.add(lhs)
                        g.P.add((lhs, tuple(rhs_syms)))

        return g

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def _state_label(self, state) -> str:
        """Generate display label for state."""
        if isinstance(state, frozenset):
            if not state:
                return "∅"
            sorted_labels = sorted(self._state_label(s) for s in state)
            return "{" + ",".join(sorted_labels) + "}"
        if isinstance(state, tuple):
            labels = [self._state_label(s) for s in state]
            return "(" + ",".join(labels) + ")"
        return str(state)

    def _get_state_id(self, state, state_to_id: dict) -> str:
        """Get or create a clean ID for a state."""
        if state not in state_to_id:
            state_to_id[state] = f"q{len(state_to_id)}"
        return state_to_id[state]

    def to_graphviz(self, filename: str = "automaton", view: bool = True) -> Digraph:
        """Generate a Graphviz visualization for this automaton."""
        type_names = {1: "DEA", 2: "NEA", 3: "NEA+ε", 4: "NEA+words", 5: "PDA"}

        dot = Digraph(
            name=type_names.get(self.type, "Automaton"),
            format="png",
            graph_attr={
                "rankdir": "LR",
                "splines": "true",
                "nodesep": "0.8",
                "ranksep": "1.2",
                "label": type_names.get(self.type, "Automaton"),
                "labelloc": "t",
                "fontsize": "14",
                "fontname": "Arial",
                "bgcolor": "white",
                "pad": "0.5",
                "dpi": "300",
            },
            node_attr={
                "shape": "circle",
                "fontsize": "14",
                "fontname": "Arial",
                "width": "0.6",
                "height": "0.6",
                "fixedsize": "true",
                "style": "filled",
                "fillcolor": "lightblue",
                "color": "black",
                "penwidth": "2",
            },
            edge_attr={
                "fontsize": "12",
                "fontname": "Arial",
                "arrowsize": "0.8",
                "penwidth": "1.5",
                "color": "black",
            },
        )

        # Create mapping from states to clean IDs
        state_to_id: Dict[Any, str] = {}

        # Invisible start arrow with consistent styling
        dot.node("__start__", shape="point", width="0.01", style="invis")

        # Add states
        for state in self.states:
            label = self._state_label(state)
            node_id = self._get_state_id(state, state_to_id)
            if state in self.accepting_states:
                dot.node(
                    node_id,
                    label=label,
                    shape="doublecircle",
                    fillcolor="lightgreen",
                    peripheries="2",
                )
            else:
                dot.node(node_id, label=label)

        # Start edge
        start_id = self._get_state_id(self.start_state, state_to_id)
        dot.edge("__start__", start_id, penwidth="2")

        # Add transitions
        if self.type == 5:  # PDA
            transitions = defaultdict(list)
            for src, inp, stk, tgt, stack_str in self.transition_relation:
                inp_label = "ε" if inp is None else str(inp)
                stk_label = "ε" if stk is None else str(stk)
                stack_label = "ε" if stack_str == "" else str(stack_str)
                label = f"{inp_label}, {stk_label} → {stack_label}"
                transitions[(src, tgt)].append(label)

            for (src, tgt), labels in transitions.items():
                label = "\n".join(labels)
                src_id = self._get_state_id(src, state_to_id)
                tgt_id = self._get_state_id(tgt, state_to_id)

                if src == tgt:
                    dot.edge(src_id, tgt_id, label=label, headport="n", tailport="n")
                else:
                    dot.edge(src_id, tgt_id, label=label)
        else:  # FA
            transitions = defaultdict(list)
            for src, sym, tgt in self.transition_relation:
                sym_label = "ε" if sym is None else str(sym)
                transitions[(src, tgt)].append(sym_label)

            for (src, tgt), symbols in transitions.items():
                label = ", ".join(sorted(symbols))
                src_id = self._get_state_id(src, state_to_id)
                tgt_id = self._get_state_id(tgt, state_to_id)

                if src == tgt:
                    dot.edge(src_id, tgt_id, label=label, headport="n", tailport="n")
                else:
                    dot.edge(src_id, tgt_id, label=label)

        dot.render(filename, view=view, cleanup=True)
        return dot


class RegularExpression:
    """
    Regular Expression class supporting conversion to/from NEA.

    Supports operations:
    - Concatenation: ab
    - Union: a+b
    - Kleene star: a*
    - Parentheses: (a+b)*
    - Empty string: ε or epsilon
    - Empty set: ∅
    """

    def __init__(self, pattern: str = ""):
        self.pattern = pattern
        self.EPSILON = "ε"
        self.EMPTY_SET = "∅"

    def __str__(self):
        return f"RegEx: {self.pattern}"

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_NEA(self) -> Automaton:
        """Convert regular expression to NEA using Thompson's construction."""
        return self._thompson_construction(self.pattern)

    # Thompson construction ------------------------------------------------

    def _thompson_construction(self, pattern: str) -> Automaton:
        """Thompson's construction algorithm for converting regex to NEA."""
        postfix = self._to_postfix(pattern)

        # Stack to hold automata fragments
        stack: List[Automaton] = []
        state_counter = [0]  # Use list to allow modification in nested function

        def new_state() -> str:
            state = f"q{state_counter[0]}"
            state_counter[0] += 1
            return state

        for token in postfix:
            if token == "+":  # Union
                nea2 = stack.pop()
                nea1 = stack.pop()

                new_start = new_state()
                new_accept = new_state()

                transitions = set(nea1.transition_relation)
                transitions.update(nea2.transition_relation)
                transitions.add((new_start, None, nea1.start_state))
                transitions.add((new_start, None, nea2.start_state))

                for acc in nea1.accepting_states:
                    transitions.add((acc, None, new_accept))
                for acc in nea2.accepting_states:
                    transitions.add((acc, None, new_accept))

                states = nea1.states | nea2.states | {new_start, new_accept}
                alphabet = nea1.alphabet | nea2.alphabet

                nea = Automaton(
                    type=3,
                    states=frozenset(states),
                    alphabet=alphabet,
                    start_state=new_start,
                    accepting_states=frozenset({new_accept}),
                    transition_relation=transitions,
                )
                stack.append(nea)

            elif token == "·":  # Concatenation
                nea2 = stack.pop()
                nea1 = stack.pop()

                transitions = set(nea1.transition_relation)
                transitions.update(nea2.transition_relation)

                for acc in nea1.accepting_states:
                    transitions.add((acc, None, nea2.start_state))

                states = nea1.states | nea2.states
                alphabet = nea1.alphabet | nea2.alphabet

                nea = Automaton(
                    type=3,
                    states=frozenset(states),
                    alphabet=alphabet,
                    start_state=nea1.start_state,
                    accepting_states=nea2.accepting_states,
                    transition_relation=transitions,
                )
                stack.append(nea)

            elif token == "*":  # Kleene star
                nea1 = stack.pop()

                new_start = new_state()
                new_accept = new_state()

                transitions = set(nea1.transition_relation)
                transitions.add((new_start, None, nea1.start_state))
                transitions.add((new_start, None, new_accept))

                for acc in nea1.accepting_states:
                    transitions.add((acc, None, nea1.start_state))
                    transitions.add((acc, None, new_accept))

                states = nea1.states | {new_start, new_accept}

                nea = Automaton(
                    type=3,
                    states=frozenset(states),
                    alphabet=nea1.alphabet,
                    start_state=new_start,
                    accepting_states=frozenset({new_accept}),
                    transition_relation=transitions,
                )
                stack.append(nea)

            elif token == self.EPSILON:  # Epsilon
                start = new_state()
                accept = new_state()

                nea = Automaton(
                    type=3,
                    states=frozenset({start, accept}),
                    alphabet=set(),
                    start_state=start,
                    accepting_states=frozenset({accept}),
                    transition_relation={(start, None, accept)},
                )
                stack.append(nea)

            elif token == self.EMPTY_SET:  # Empty set
                start = new_state()

                nea = Automaton(
                    type=3,
                    states=frozenset({start}),
                    alphabet=set(),
                    start_state=start,
                    accepting_states=frozenset(),
                    transition_relation=set(),
                )
                stack.append(nea)

            else:  # Single symbol
                start = new_state()
                accept = new_state()

                nea = Automaton(
                    type=3,
                    states=frozenset({start, accept}),
                    alphabet={token},
                    start_state=start,
                    accepting_states=frozenset({accept}),
                    transition_relation={(start, token, accept)},
                )
                stack.append(nea)

        return stack[0] if stack else Automaton(type=3)

    # Shunting-yard to postfix ----------------------------------------------

    def _to_postfix(self, pattern: str) -> List[str]:
        # Add explicit concatenation operators
        pattern = self._add_concat_operator(pattern)

        output: List[str] = []
        stack: List[str] = []

        precedence = {"+": 1, "·": 2, "*": 3}

        for char in pattern:
            if char == "(":
                stack.append(char)
            elif char == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if stack:
                    stack.pop()  # Remove "("
            elif char in precedence:
                while (
                    stack
                    and stack[-1] != "("
                    and stack[-1] in precedence
                    and precedence[stack[-1]] >= precedence[char]
                ):
                    output.append(stack.pop())
                stack.append(char)
            else:
                output.append(char)

        while stack:
            output.append(stack.pop())

        return output

    def _add_concat_operator(self, pattern: str) -> str:
        result: List[str] = []

        for i, char in enumerate(pattern):
            result.append(char)

            if i < len(pattern) - 1:
                next_char = pattern[i + 1]

                if (
                    (char not in "(+" and next_char not in ")+*")
                    or (char == "*" and next_char == "(")
                    or (char == ")" and next_char == "(")
                ):
                    result.append("·")

        return "".join(result)

    # Constructors ----------------------------------------------------------

    @classmethod
    def from_string(cls, pattern: str) -> "RegularExpression":
        """Create a RegularExpression from a string pattern."""
        return cls(pattern)

    @classmethod
    def from_file(cls, filename: str) -> "RegularExpression":
        """Load a regular expression from a file."""
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return cls(content)

