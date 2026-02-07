from collections import defaultdict
from typing_extensions import *
import os
import math
from itertools import combinations
from graphviz import Digraph
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
import re
import tempfile



def powerset_noemptyset(s):
    s = list(s)
    return frozenset({
        frozenset(c)
        for r in range(1, len(s) + 1)
        for c in combinations(s, r)
    })

@dataclass
class Automaton:

    type: int = 1  # 1=DEA, 2=NEA, 3=NEA+epsilon, 4=NEA+words
    states: frozenset = field(default_factory=frozenset)
    alphabet: Set[str] = field(default_factory=set)
    start_state: Any = None
    
    # Transition relation: Set of (source_state, symbol, target_state) tuples
    # Works for all automaton types
    transition_relation: Set[Tuple[Any, Any, Any]] = field(default_factory=set)
    
    accepting_states: frozenset = field(default_factory=frozenset)
    indices: Dict[Any, int] = field(default_factory=dict)

    @staticmethod
    def load_from_file(file_path: str) -> List['Automaton']:
        """
        Load one or more automata from a file.
        Returns a list of Automaton objects.
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split by delimiter (---) to support multiple automata
        automaton_blocks = content.split('---')
        
        automata = []
        for block in automaton_blocks:
            block = block.strip()
            if not block:
                continue
            automaton = Automaton._parse_block(block)
            automata.append(automaton)
        
        return automata
    
    @staticmethod
    def _parse_block(block: str) -> 'Automaton':
        """Parse a single automaton block."""
        states = set()
        alphabet = set()
        start_state = None
        accepting_states = set()
        transitions = set()
        automaton_type = 1  # default DEA
        
        lines = block.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse type
            elif line.startswith('type:'):
                t = line[5:].strip()
                if t.isdigit():
                    automaton_type = int(t)
                else:
                    # Support readable type names
                    type_map = {
                        'dfa': 1, 'dea': 1,
                        'nfa': 2, 'nea': 2,
                        'enfa': 3, 'nfa-e': 3, 'epsilon': 3,
                        'nfaw': 4, 'word': 4
                    }
                    automaton_type = type_map.get(t.lower(), 1)
            
            # Parse alphabet
            elif line.startswith('alphabet:'):
                symbols = line[9:].strip().split()
                alphabet.update(symbols)
            
            # Parse states
            elif line.startswith('states:'):
                state_list = line[7:].strip().split()
                states.update(state_list)
            
            # Parse start state
            elif line.startswith('start:'):
                start_state = line[6:].strip()
            
            # Parse accepting states
            elif line.startswith('accept:'):
                acc_list = line[7:].strip().split()
                accepting_states.update(acc_list)
            
            # Parse transitions (format: q0 a q1 or q0 -> a -> q1)
            elif '->' in line:
                # Support arrow notation: q0 -> a -> q1
                parts = [p.strip() for p in line.split('->')]
                if len(parts) == 3:
                    src, symbol, tgt = parts
                elif len(parts) == 2:
                    # Format: q0 -> a (epsilon transition to implicit state)
                    src, symbol = parts
                    tgt = None
                else:
                    continue
                
                # Handle epsilon symbol
                if symbol.lower() in ['eps', 'epsilon', 'ε', 'e', '']:
                    symbol = None
                
                transitions.add((src, symbol, tgt))
            
            elif len(line.split()) >= 3:
                # Simple format: q0 a q1
                parts = line.split()
                src = parts[0]
                symbol = parts[1]
                tgt = parts[2]
                
                # Handle epsilon symbol
                if symbol.lower() in ['eps', 'epsilon', 'ε', 'e']:
                    symbol = None
                
                transitions.add((src, symbol, tgt))
        
        # Build indices
        indices = {s: i for i, s in enumerate(sorted(states))}
        
        return Automaton(
            type=automaton_type,
            states=frozenset(states),
            alphabet=alphabet,
            start_state=start_state,
            accepting_states=frozenset(accepting_states),
            transition_relation=transitions,
            indices=indices,
        )
    
    def __post_init__(self):
        """Validate the automaton type and structure."""
        if self.type not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid automaton type: {self.type}. Must be 1, 2, 3, or 4.")
    
    def _get_targets(self, state: Any, symbol: Any) -> frozenset:
        """Get all target states for a given (state, symbol) pair."""
        return frozenset({
            target for (src, sym, target) in self.transition_relation
            if src == state and sym == symbol
        })
    
    def _get_transition_dict(self) -> Dict[Tuple[Any, Any], frozenset]:
        """Convert transition relation to dictionary format for convenience."""
        result = defaultdict(set)
        for src, sym, tgt in self.transition_relation:
            result[(src, sym)].add(tgt)
        return {k: frozenset(v) for k, v in result.items()}
    
    # ==================== DEA-specific methods ====================
    
    def complement(self) -> 'Automaton':
        """Compute complement (only for DEA)."""
        if self.type != 1:
            raise ValueError("Complement operation only defined for DEA (type 1)")
        
        return Automaton(
            type=1,
            start_state=self.start_state,
            alphabet=self.alphabet,
            transition_relation=self.transition_relation,
            states=self.states,
            accepting_states=self.states.difference(self.accepting_states)
        )
    
    def intersection(self, other: Self) -> 'Automaton':
        """Compute intersection (only for DEA)."""
        if self.type != 1 or other.type != 1:
            raise ValueError("Intersection operation only defined for DEA (type 1)")
        
        new_states = frozenset({
            (S, S2) 
            for S in self.states 
            for S2 in other.states
        })
        
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
            accepting_states=frozenset({
                (S, S2) 
                for S in self.accepting_states
                for S2 in other.accepting_states
            }),
            start_state=(self.start_state, other.start_state)
        )
    
    def compute_equivalence_classes(self) -> Set[frozenset]:
        """
        Compute equivalence classes by iterative refinement (only for DEA).
        This is the same as finding the largest k where ~_k = ~_(k+1)
        """
        if self.type != 1:
            raise ValueError("Equivalence class computation only defined for DEA (type 1)")
        
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

    def to_grammar(self) -> 'Grammar':
        grammar = Grammar()
        for state in self.states:
            grammar.add_non_terminal(state)
        for (A, a, B) in self.transition_relation:
            grammar.add_production(A, [a,B])
        for A in self.accepting_states:
            grammar.add_production(A, grammar.EPSILON)

        grammar.set_start_symbol(self.start_state)

        return grammar



    def minimize(self) -> 'Automaton':
        """Minimize the automaton (only for DEA)."""
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
        
        new_accepting = frozenset({
            eq_class for eq_class in equiv_classes
            if any(state in self.accepting_states for state in eq_class)
        })
        
        return Automaton(
            type=1,
            states=new_states,
            alphabet=self.alphabet,
            start_state=new_start,
            transition_relation=new_relation,
            accepting_states=new_accepting
        )
    
    # ==================== NEA-specific methods ====================
    
    def to_DEA(self) -> 'Automaton':
        """Convert NEA (types 2, 3, 4) to DEA using powerset construction."""
        if self.type == 1:
            return self  # Already a DEA
        
        if self.type == 3:
            # First eliminate epsilon transitions
            return self._eliminate_epsilon().to_DEA()
        
        if self.type == 4:
            # First eliminate word transitions
            return self._eliminate_word_transitions().to_DEA()
        

        new_states = powerset_noemptyset(self.states)
        new_start = frozenset({self.start_state})
        
        # Build transition dictionary for easier lookup
        trans_dict = self._get_transition_dict()
        
        # Build new transitions
        new_relation = set()
        for S in new_states:
            for a in self.alphabet:
                target_set = set()
                for q in S:
                    target_set.update(trans_dict.get((q, a), frozenset()))
                if target_set:
                    new_relation.add((S, a, frozenset(target_set)))

        accepting = frozenset({
            S for S in new_states 
            if len(S.intersection(self.accepting_states)) != 0
        })

        return Automaton(
            type=1,
            states=new_states,
            accepting_states=accepting,
            start_state=new_start,
            transition_relation=new_relation,
            alphabet=self.alphabet
        )
    
    def _eliminate_epsilon(self) -> 'Automaton':
        """Eliminate epsilon transitions (convert type 3 to type 2)."""
        if self.type != 3:
            return self
        
        # Build transition lookup
        trans_dict = self._get_transition_dict()
        
        # Compute epsilon closure for each state
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
        
        # Build new transition relation without epsilon
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
        
        # Update accepting states to include states with epsilon-path to accepting
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
            accepting_states=frozenset(new_accepting)
        )
    
    def _eliminate_word_transitions(self) -> 'Automaton':
        """Eliminate word transitions (convert type 4 to type 2)."""
        if self.type != 4:
            return self
        
        # Create new intermediate states for word transitions
        new_states = set(self.states)
        new_relation = set()
        state_counter = 0
        
        for src, word, tgt in self.transition_relation:
            if word is None or len(word) == 0:  # Epsilon
                new_relation.add((src, None, tgt))
            elif len(word) == 1:  # Single symbol
                new_relation.add((src, word, tgt))
            else:  # Multi-symbol word
                # Create intermediate states
                prev_state = src
                for i, symbol in enumerate(word[:-1]):
                    intermediate = f"_q{state_counter}"
                    state_counter += 1
                    new_states.add(intermediate)
                    new_relation.add((prev_state, symbol, intermediate))
                    prev_state = intermediate
                # Last transition
                new_relation.add((prev_state, word[-1], tgt))
        
        return Automaton(
            type=3,  # Result has epsilon transitions
            states=frozenset(new_states),
            alphabet=self.alphabet,
            start_state=self.start_state,
            transition_relation=new_relation,
            accepting_states=self.accepting_states
        )
    
    # ==================== Visualization ====================
    
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
        type_names = {1: "DEA", 2: "NEA", 3: "NEA+ε", 4: "NEA+words"}
        
        dot = Digraph(
            name=type_names.get(self.type, "Automaton"),
            format="png",
            graph_attr={
                "rankdir": "LR",
                "splines": "true",  # Changed from "true" for smoother edges
                "nodesep": "0.8",     # Increased spacing between nodes
                "ranksep": "1.2",     # Increased spacing between ranks
                "label": type_names.get(self.type, "Automaton"),
                "labelloc": "t",
                "fontsize": "14",     # Larger title font
                "fontname": "Arial",  # Consistent font
                "bgcolor": "white",   # Explicit background
                "pad": "0.5",         # Padding around graph
                "dpi": "300"          # Higher resolution
            },
            node_attr={
                "shape": "circle",
                "fontsize": "14",     # Slightly larger font
                "fontname": "Arial",
                "width": "0.6",       # Fixed width for consistency
                "height": "0.6",      # Fixed height for consistency
                "fixedsize": "true",  # CRITICAL: prevents nodes from resizing
                "style": "filled",
                "fillcolor": "lightblue",
                "color": "black",
                "penwidth": "2"       # Thicker border
            },
            edge_attr={
                "fontsize": "12",
                "fontname": "Arial",
                "arrowsize": "0.8",   # Consistent arrow size
                "penwidth": "1.5",    # Thicker edges
                "color": "black"
            }
        )
        
        # Create mapping from states to clean IDs
        state_to_id = {}
        
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
                    fillcolor="lightgreen",  # Different color for accepting states
                    peripheries="2"
                )
            else:
                dot.node(node_id, label=label)
        
        # Start edge
        start_id = self._get_state_id(self.start_state, state_to_id)
        dot.edge("__start__", start_id, penwidth="2")
        
        # Add transitions - group by (src, tgt) for cleaner labels
        transitions = defaultdict(list)
        for src, sym, tgt in self.transition_relation:
            sym_label = "ε" if sym is None else str(sym)
            transitions[(src, tgt)].append(sym_label)
        
        for (src, tgt), symbols in transitions.items():
            label = ", ".join(sorted(symbols))  # Added space after comma
            src_id = self._get_state_id(src, state_to_id)
            tgt_id = self._get_state_id(tgt, state_to_id)
            
            # Self-loops get special treatment
            if src == tgt:
                dot.edge(src_id, tgt_id, label=label, headport="n", tailport="n")
            else:
                dot.edge(src_id, tgt_id, label=label)
        
        dot.render(filename, view=view, cleanup=True)
        return dot

class GrammarType(Enum):
    """Chomsky hierarchy of formal grammars"""
    TYPE_0 = 0  # Unrestricted grammar
    TYPE_1 = 1  # Context-sensitive grammar
    TYPE_2 = 2  # Context-free grammar (CFG)
    TYPE_3 = 3  # Regular grammar


class Grammar:
    """
    General Grammar class supporting Chomsky hierarchy (Type-0 through Type-3)
    
    A grammar is defined by:
    - N: Set of non-terminal symbols
    - Sigma: Set of terminal symbols (alphabet)
    - P: Set of production rules
    - S: Start symbol (S ∈ N)
    - type: Grammar type (Type-0, Type-1, Type-2, or Type-3)
    
    Production rules format:
    - Internal: (lhs: str, rhs: List[str]) where each element is a symbol
    - External: Can accept strings which are converted to symbol lists
    
    For single-character symbols: "aSb" -> ['a', 'S', 'b']
    For multi-character symbols: Use list directly: ['X1', 'N2']
    """
    
    def __init__(self, grammar_type: GrammarType = GrammarType.TYPE_2):
        self.type: GrammarType = grammar_type
        self.N: Set[str] = set()  # Non-terminals
        self.Sigma: Set[str] = set()  # Terminals
        self.P: Set[Tuple[str, Tuple[str, ...]]] = set()  # Productions (lhs, tuple of rhs symbols)
        self.S: str = ""  # Start symbol
        self.EPSILON = 'ε'  # Epsilon symbol
    
    def add_non_terminal(self, symbol: str):
        """Add a non-terminal symbol"""
        self.N.add(symbol)
    
    def add_terminal(self, symbol: str):
        """Add a terminal symbol"""
        if symbol == self.EPSILON:
            return  # Don't add epsilon to alphabet
        self.Sigma.add(symbol)

    def detect_grammar_type(self) -> GrammarType:
        """
        Detect the grammar type using generalized Type-3 definition:
        A → uB or A → u, where u is any string of terminals.
        """
        is_type3 = True
        is_type2 = True
        is_type1 = True
        grammar = self

        for lhs, rhs in grammar.P:
            # RHS as string for easier checks
            rhs_str = ''.join(rhs)

            # --- Type 3 check (generalized regular) ---
            if not (len(lhs) == 1 and lhs in grammar.N):
                is_type3 = False
            else:
                if rhs != (grammar.EPSILON,):
                    # Split rhs into terminals and optional last non-terminal
                    if len(rhs) >= 1 and rhs[-1] in grammar.N:
                        u, B = rhs[:-1], rhs[-1]  # u = terminals, B = non-terminal
                        if any(sym not in grammar.Sigma for sym in u):
                            is_type3 = False
                    else:
                        # All symbols must be terminals
                        if any(sym not in grammar.Sigma for sym in rhs):
                            is_type3 = False

            # --- Type 2 check (context-free) ---
            if not (len(lhs) == 1 and lhs in grammar.N):
                is_type2 = False

            # --- Type 1 check (context-sensitive) ---
            if rhs != (grammar.EPSILON,):
                if len(rhs) < len(lhs):
                    is_type1 = False
            else:
                if lhs != grammar.S:  # ε allowed only for start symbol
                    is_type1 = False

        # Return the highest type that holds
        if is_type3:
            return GrammarType.TYPE_3
        elif is_type2:
            return GrammarType.TYPE_2
        elif is_type1:
            return GrammarType.TYPE_1
        else:
            return GrammarType.TYPE_0

    def add_production(self, lhs: str, rhs: Union[str, List[str]]):
        """
        Add a production rule α -> β
        Validates production based on grammar type
        
        Args:
            lhs: Left-hand side (single non-terminal for Type-2/3)
            rhs: Right-hand side as string (for single-char symbols) or list of symbols
                 String "aSb" -> ['a', 'S', 'b']
                 List ['X1', 'N2'] -> ['X1', 'N2']
        """
        # Convert rhs to list if it's a string
        if isinstance(rhs, str):
            if rhs == self.EPSILON:
                rhs_list = [self.EPSILON]
            else:
                rhs_list = list(rhs)  # Each character is a symbol
        else:
            rhs_list = rhs
        
        if not self._validate_production(lhs, rhs_list):
            raise ValueError(f"Invalid production {lhs} -> {rhs_list} for {self.type.name}")
        
        # Store as tuple for hashability
        self.P.add((lhs, tuple(rhs_list)))
        
        # Auto-add non-terminals from lhs (for Type-2 and Type-3)
        if self.type in [GrammarType.TYPE_2, GrammarType.TYPE_3]:
            if lhs not in self.N:
                self.N.add(lhs)
    
    def _validate_production(self, lhs: str, rhs: List[str]) -> bool:
        """Validate production rule based on grammar type"""
        if self.type == GrammarType.TYPE_0:
            # Unrestricted: α -> β where α contains at least one non-terminal
            return len(lhs) > 0 and any(symbol in self.N for symbol in lhs)
        
        elif self.type == GrammarType.TYPE_1:
            # Context-sensitive: |α| ≤ |β| (except for S -> ε if S doesn't appear in rhs)
            if len(rhs) == 1 and rhs[0] == self.EPSILON:
                return lhs == self.S
            return len(lhs) <= len(rhs)
        
        elif self.type == GrammarType.TYPE_2:
            # Context-free: A -> α where A is a single non-terminal
            return len(lhs) == 1 and lhs in self.N
        
        elif self.type == GrammarType.TYPE_3:
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
    
    def set_start_symbol(self, symbol: str):
        """Set the start symbol"""
        self.S = symbol
        if symbol not in self.N:
            self.N.add(symbol)
    
    def __str__(self):
        """String representation of the grammar"""
        result = f"Grammar Type: {self.type.name}\n"
        result += f"  Non-terminals: {{{', '.join(sorted(self.N))}}}\n"
        result += f"  Terminals: {{{', '.join(sorted(self.Sigma))}}}\n"
        result += f"  Start symbol: {self.S}\n"
        result += "  Productions:\n"
        
        # Group productions by lhs
        prod_dict: Dict[str, List[str]] = {}
        for lhs, rhs_tuple in sorted(self.P):
            # Convert tuple back to string for display
            rhs_str = ''.join(rhs_tuple)
            if lhs not in prod_dict:
                prod_dict[lhs] = []
            prod_dict[lhs].append(rhs_str)
        
        for lhs in sorted(prod_dict.keys()):
            result += f"    {lhs} -> {' | '.join(prod_dict[lhs])}\n"
        
        return result
    
    def copy(self):
        """Create a deep copy of the grammar"""
        new_grammar = Grammar(self.type)
        new_grammar.N = deepcopy(self.N)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.P = deepcopy(self.P)
        new_grammar.S = self.S
        new_grammar.EPSILON = self.EPSILON
        return new_grammar
    
    def convert_type(self, target_type: GrammarType):
        """
        Convert grammar to a different type (if possible)
        Note: Not all conversions are possible or preserve the language
        """
        if self.type == target_type:
            return self.copy()
        
        # For now, only implement Type-2 to CNF as an example
        if self.type == GrammarType.TYPE_2 and target_type == GrammarType.TYPE_2:
            return self.copy()
        
        raise NotImplementedError(f"Conversion from {self.type.name} to {target_type.name} not implemented")
    
    @classmethod
    def from_file(cls, filename: str, grammar_type: GrammarType = GrammarType.TYPE_2):
        """
        Parse a grammar from a file.
        
        Simplified format - just list productions:
        ```
        S -> aSb
        S -> ε
        A -> aA | b
        ```
        
        Auto-detection rules:
        - Start symbol: 'S' if it exists, otherwise the first non-terminal
        - Non-terminals: All single uppercase letters (A-Z)
        - Terminals: Everything else (lowercase, digits, symbols)
        - Epsilon: ε, epsilon, λ, or empty string
        
        Optional configuration (if needed):
        ```
        TYPE: 2
        START: E
        
        E -> E+T | T
        T -> T*F | F
        ```
        
        Args:
            filename: Path to the grammar file
            grammar_type: Default grammar type if not specified in file
        
        Returns:
            Grammar object parsed from the file
        """
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return cls.from_string(content, grammar_type)
    
    @classmethod
    def from_string(cls, content: str, grammar_type: GrammarType = GrammarType.TYPE_2):
        """
        Parse a grammar from a string.
        
        Auto-detects everything from productions:
        - Non-terminals: Single uppercase letters (A-Z)
        - Terminals: Everything else
        - Start symbol: 'S' if present, otherwise first non-terminal found
        
        Args:
            content: Grammar specification as a string
            grammar_type: Default grammar type if not specified in content
        
        Returns:
            Grammar object parsed from the string
        """
        lines = content.strip().split('\n')
        
        # Check if it's BNF format
        is_bnf_format = any('::=' in line for line in lines)
        
        if is_bnf_format:
            return cls._parse_bnf_format(content, grammar_type)
        else:
            return cls._parse_simplified_format(content, grammar_type)
    
    @classmethod
    def _parse_simplified_format(cls, content: str, default_type: GrammarType):
        """
        Parse simplified format - auto-detect everything from productions.
        
        Rules:
        - Single uppercase letters (A-Z) are non-terminals
        - Everything else is a terminal
        - Start symbol is 'S' if it exists, otherwise first non-terminal
        """
        lines = content.strip().split('\n')
        
        grammar_type = default_type
        specified_start = None
        productions = []
        
        # First pass: collect configuration and productions
        for line in lines:
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            
            if not line:
                continue
            
            # Check for optional configuration
            if line.startswith('TYPE:'):
                type_num = int(line.split(':')[1].strip())
                grammar_type = GrammarType(type_num)
                continue
            
            if line.startswith('START:'):
                specified_start = line.split(':')[1].strip()
                continue
            
            # Skip old-style section headers
            if line in ['PRODUCTIONS:', 'TERMINALS:', 'NON_TERMINALS:', 'NONTERMINALS:']:
                continue
            
            if line.startswith('TERMINALS:') or line.startswith('NON_TERMINALS:') or line.startswith('NONTERMINALS:') or line.startswith('EPSILON:'):
                continue
            
            # Parse productions (both -> and ::=)
            if '->' in line or '→' in line or '::=' in line:
                # Normalize arrows
                line = line.replace('→', '->').replace('::=', '->')
                productions.append(line)
        
        # Create grammar
        g = cls(grammar_type)
        
        # Second pass: parse productions and auto-detect symbols
        all_lhs = set()
        all_symbols = set()
        
        for prod_line in productions:
            if '->' in prod_line:
                parts = prod_line.split('->')
                lhs = parts[0].strip()
                rhs_alternatives = parts[1].strip()
                
                # Remove BNF brackets from lhs if present
                if lhs.startswith('<') and lhs.endswith('>'):
                    lhs = lhs[1:-1]
                
                all_lhs.add(lhs)
                
                # Handle multiple alternatives separated by |
                for rhs in rhs_alternatives.split('|'):
                    rhs = rhs.strip()
                    
                    # Remove BNF brackets and collect all symbols
                    i = 0
                    cleaned_rhs = ""
                    while i < len(rhs):
                        if rhs[i] == '<':
                            # Find matching >
                            end = rhs.index('>', i)
                            symbol = rhs[i+1:end]
                            all_symbols.add(symbol)
                            cleaned_rhs += symbol
                            i = end + 1
                        elif rhs[i].isspace():
                            i += 1
                        else:
                            all_symbols.add(rhs[i])
                            cleaned_rhs += rhs[i]
                            i += 1
                    
                    # Collect all symbols from rhs
                    for char in rhs:
                        if char not in '<>| \t':
                            all_symbols.add(char)
        
        # Auto-detect non-terminals: single uppercase letters
        non_terminals = set()
        for symbol in all_symbols | all_lhs:
            if len(symbol) == 1 and symbol.isupper():
                non_terminals.add(symbol)
        
        # Auto-detect terminals: everything else (except epsilon variants)
        terminals = set()
        epsilon_variants = {'ε', 'epsilon', 'EPSILON', 'λ', ''}
        for symbol in all_symbols:
            if symbol not in non_terminals and symbol not in epsilon_variants:
                terminals.add(symbol)
        
        # Add symbols to grammar
        for nt in non_terminals:
            g.add_non_terminal(nt)
        
        for t in terminals:
            g.add_terminal(t)
        
        # Determine start symbol
        if specified_start:
            start_symbol = specified_start
        elif 'S' in non_terminals:
            start_symbol = 'S'
        elif all_lhs:
            # Use first non-terminal that appears on LHS
            start_symbol = sorted(all_lhs)[0]
        else:
            raise ValueError("No productions found or unable to determine start symbol")
        
        g.set_start_symbol(start_symbol)
        
        # Third pass: add productions with normalized symbols
        for prod_line in productions:
            if '->' in prod_line:
                parts = prod_line.split('->')
                lhs = parts[0].strip()
                rhs_alternatives = parts[1].strip()
                
                # Remove BNF brackets from lhs
                if lhs.startswith('<') and lhs.endswith('>'):
                    lhs = lhs[1:-1]
                
                # Handle multiple alternatives
                for rhs in rhs_alternatives.split('|'):
                    rhs = rhs.strip()
                    
                    # Remove BNF brackets from rhs
                    cleaned_rhs = ""
                    i = 0
                    while i < len(rhs):
                        if rhs[i] == '<':
                            end = rhs.index('>', i)
                            cleaned_rhs += rhs[i+1:end]
                            i = end + 1
                        elif rhs[i].isspace():
                            i += 1
                        else:
                            cleaned_rhs += rhs[i]
                            i += 1
                    
                    # Replace epsilon representations
                    if cleaned_rhs in epsilon_variants:
                        cleaned_rhs = g.EPSILON
                    
                    try:
                        g.add_production(lhs, cleaned_rhs)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid production '{lhs} -> {cleaned_rhs}': {e}")

        g.type = g.detect_grammar_type()
        return g
    
    @classmethod
    def _parse_simple_format(cls, content: str, default_type: GrammarType):
        """
        Legacy parser for old explicit format.
        Kept for backward compatibility.
        """
        return cls._parse_simplified_format(content, default_type)
    
    @classmethod
    def _parse_bnf_format(cls, content: str, default_type: GrammarType):
        """
        Parse BNF-like format grammar specification.
        
        Example:
        <S> ::= a<S>b | ε
        <A> ::= a<A> | b
        """
        lines = content.strip().split('\n')
        
        g = cls(default_type)
        start_symbol = None
        
        for line in lines:
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            
            if not line:
                continue
            
            # Parse BNF production
            if '::=' in line:
                parts = line.split('::=')
                lhs = parts[0].strip()
                rhs_alternatives = parts[1].strip()
                
                # Extract non-terminal from <...>
                if lhs.startswith('<') and lhs.endswith('>'):
                    lhs = lhs[1:-1]
                
                # Set first non-terminal as start symbol
                if start_symbol is None:
                    start_symbol = lhs
                    g.set_start_symbol(start_symbol)
                
                g.add_non_terminal(lhs)
                
                # Parse alternatives
                for rhs in rhs_alternatives.split('|'):
                    rhs = rhs.strip()
                    
                    # Convert <...> to plain non-terminals and extract terminals
                    converted_rhs = ""
                    i = 0
                    while i < len(rhs):
                        if rhs[i] == '<':
                            # Find matching >
                            end = rhs.index('>', i)
                            nt = rhs[i+1:end]
                            g.add_non_terminal(nt)
                            converted_rhs += nt
                            i = end + 1
                        elif rhs[i] in ['ε', 'λ']:
                            converted_rhs += g.EPSILON
                            i += 1
                        elif rhs[i].isspace():
                            i += 1
                        else:
                            # Terminal symbol
                            g.add_terminal(rhs[i])
                            converted_rhs += rhs[i]
                            i += 1
                    
                    # Handle epsilon
                    if converted_rhs in ['ε', 'epsilon', 'λ', '']:
                        converted_rhs = g.EPSILON
                    
                    try:
                        g.add_production(lhs, converted_rhs)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid production '{lhs} -> {converted_rhs}': {e}")
        
        return g
    
    # =========================================================================
    # Type-2 (Context-Free Grammar) specific methods
    # =========================================================================
    
    def remove_non_terminating(self):
        """
        Remove all non-terminating non-terminal symbols.
        Only applicable to Type-2 grammars.
        
        A non-terminal A is terminating if there exists a derivation A =>* w
        where w ∈ Sigma* (i.e., A can derive a string of only terminals).
        
        Returns a new grammar without non-terminating symbols.
        """
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_non_terminating only works for Type-2 grammars")
        
        # Find all terminating non-terminals using fixed-point iteration
        terminating = set()
        changed = True
        
        while changed:
            changed = False
            for lhs, rhs_tuple in self.P:
                if lhs not in terminating:
                    # Check if all symbols in rhs are terminating or terminals
                    all_terminating = True
                    for symbol in rhs_tuple:
                        if symbol == self.EPSILON:
                            continue
                        # If it's a non-terminal and not yet terminating, fail
                        if symbol in self.N and symbol not in terminating:
                            all_terminating = False
                            break
                        # If it's not in N and not in Sigma, it's invalid
                        if symbol not in self.N and symbol not in self.Sigma and symbol != self.EPSILON:
                            all_terminating = False
                            break
                    
                    if all_terminating:
                        terminating.add(lhs)
                        changed = True
        
        # Create new grammar with only terminating symbols
        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.EPSILON = self.EPSILON
        
        # Check if start symbol is terminating
        if self.S not in terminating:
            return new_grammar
        
        new_grammar.S = self.S
        
        # Add only terminating non-terminals
        for nt in terminating:
            new_grammar.N.add(nt)
        
        # Add only productions where lhs is terminating and all non-terminals in rhs are terminating
        for lhs, rhs_tuple in self.P:
            if lhs in terminating:
                valid = True
                for symbol in rhs_tuple:
                    if symbol in self.N and symbol not in terminating:
                        valid = False
                        break
                if valid:
                    new_grammar.P.add((lhs, rhs_tuple))
        
        return new_grammar
    
    def remove_unreachable(self):
        """
        Remove all unreachable non-terminal symbols.
        Only applicable to Type-2 grammars.
        
        A non-terminal A is reachable if there exists a derivation S =>* αAβ
        where α, β ∈ (N ∪ Sigma)*.
        
        Returns a new grammar without unreachable symbols.
        """
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_unreachable only works for Type-2 grammars")
        
        # Find all reachable symbols using BFS
        reachable = {self.S}
        queue = [self.S]
        
        while queue:
            current = queue.pop(0)
            
            # Look at all productions from current symbol
            for lhs, rhs_tuple in self.P:
                if lhs == current:
                    # Add all symbols in rhs to reachable
                    for symbol in rhs_tuple:
                        if symbol in self.N and symbol not in reachable:
                            reachable.add(symbol)
                            queue.append(symbol)
        
        
        # Create new grammar with only reachable symbols
        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.S = self.S
        new_grammar.EPSILON = self.EPSILON
        
        # Add only reachable non-terminals
        for nt in reachable:
            new_grammar.N.add(nt)
        
        # Add only productions involving reachable non-terminals
        for lhs, rhs_tuple in self.P:
            if lhs in reachable:
                # Also check if all non-terminals in rhs are reachable
                valid = True
                for symbol in rhs_tuple:
                    if symbol in self.N and symbol not in reachable:
                        valid = False
                        break
                if valid:
                    new_grammar.P.add((lhs, rhs_tuple))
        
        return new_grammar
    
    def remove_epsilon_productions(self):
        """
        Remove all ε-productions (A -> ε).
        Only applicable to Type-2 grammars.
        
        Algorithm:
        1. Find all nullable non-terminals (can derive ε)
        2. For each production, create new productions by removing nullable symbols
        3. If start symbol is nullable, create new start symbol S0 with S0 -> S | ε
        4. Remove all ε-productions except from the new start symbol
        
        Returns a new grammar without ε-productions.
        """
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_epsilon_productions only works for Type-2 grammars")
        
        nullable = set()
        changed = True
        
        while changed:
            changed = False
            for lhs, rhs_tuple in self.P:
                if lhs not in nullable:
                    if len(rhs_tuple) == 1 and rhs_tuple[0] == self.EPSILON:
                        nullable.add(lhs)
                        changed = True
                    else:
                        # Check if all symbols in rhs are nullable
                        all_nullable = True
                        for symbol in rhs_tuple:
                            if symbol not in nullable:
                                all_nullable = False
                                break
                        if all_nullable and len(rhs_tuple) > 0:
                            nullable.add(lhs)
                            changed = True
        
        # Create new grammar
        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(self.Sigma)
        new_grammar.EPSILON = self.EPSILON
        
        # Check if start symbol is nullable
        start_is_nullable = self.S in nullable
        
        if start_is_nullable:
            # Create new start symbol
            new_start = 'S0'
            # Make sure it's unique
            while new_start in self.N:
                # Extract number and increment
                if new_start[-1].isdigit():
                    num = int(new_start[1:]) + 1
                    new_start = f'S{num}'
                else:
                    new_start = new_start + '0'
            
            new_grammar.S = new_start
            new_grammar.N.add(new_start)
            new_grammar.P.add((new_start, (self.S,)))
            new_grammar.P.add((new_start, (self.EPSILON,)))
        else:
            new_grammar.S = self.S
        
        # Add all original non-terminals
        for nt in self.N:
            new_grammar.N.add(nt)
        
        # For each production, generate all combinations by removing nullable symbols
        for lhs, rhs_tuple in self.P:
            if len(rhs_tuple) == 1 and rhs_tuple[0] == self.EPSILON:
                # Skip epsilon productions - they're handled by new start symbol if needed
                continue
            
            # Find positions of nullable symbols in rhs
            nullable_positions = []
            for i, symbol in enumerate(rhs_tuple):
                if symbol in nullable:
                    nullable_positions.append(i)
            
            # Generate all subsets of nullable positions
            num_nullable = len(nullable_positions)
            for mask in range(1 << num_nullable):  # 2^n combinations
                new_rhs = []
                for i, symbol in enumerate(rhs_tuple):
                    # Check if this position should be removed
                    if i in nullable_positions:
                        pos_idx = nullable_positions.index(i)
                        if mask & (1 << pos_idx):
                            continue  # Remove this symbol
                    new_rhs.append(symbol)
                
                if new_rhs:  # Don't add empty productions
                    new_grammar.P.add((lhs, tuple(new_rhs)))
        
        
        return new_grammar
    
    def remove_unit_productions(self):
        """
        Remove all unit productions (A -> B where B is a non-terminal).
        Only applicable to Type-2 grammars.
        
        Algorithm:
        1. Find all unit pairs (A, B) where A =>* B via unit productions
        2. For each unit pair (A, B), add A -> α for all B -> α where α is not a single non-terminal
        3. Remove all unit productions
        
        Returns a new grammar without unit productions.
        """
        if self.type != GrammarType.TYPE_2:
            raise ValueError("remove_unit_productions only works for Type-2 grammars")
        
        
        # Find all unit pairs using transitive closure
        unit_pairs = set()
        
        # Initialize with direct unit productions
        for lhs, rhs_tuple in self.P:
            if len(rhs_tuple) == 1 and rhs_tuple[0] in self.N:
                unit_pairs.add((lhs, rhs_tuple[0]))
        
        # Add reflexive pairs
        for nt in self.N:
            unit_pairs.add((nt, nt))
        
        changed = True
        while changed:
            changed = False
            new_pairs = set()
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
        
        # Add non-unit productions
        for lhs, rhs_tuple in self.P:
            # Skip unit productions
            if len(rhs_tuple) == 1 and rhs_tuple[0] in self.N:
                continue
            new_grammar.P.add((lhs, rhs_tuple))
        
        # For each unit pair (A, B), add A -> α for all non-unit B -> α
        for a, b in unit_pairs:
            for lhs, rhs_tuple in self.P:
                if lhs == b:
                    # Skip unit productions
                    if len(rhs_tuple) == 1 and rhs_tuple[0] in self.N:
                        continue
                    new_grammar.P.add((a, rhs_tuple))
        
        
        return new_grammar
    def to_NEA(self) -> 'Automaton':
        if self.type != GrammarType.TYPE_3:
            raise ValueError("to NEA only works for Type-3 grammars")

        nea = Automaton()
        nea.type = 4
        new_symbol = "F"
        relation = set()
        for (A, w) in self.P:
            (*u, B) = w
            if B in self.N:
                relation.update({(A, "".join(u), B)})
            else:
                relation.update({ (A, "".join(w), new_symbol) })
        print(relation)
        nea.accepting_states = frozenset({new_symbol})
        nea.transition_relation = relation
        nea.start_state = self.S
        nea.states = frozenset(self.N.union(new_symbol))
        nea.alphabet = self.Sigma
        return nea
    
    def to_chomsky_normal_form(self):

        if self.type != GrammarType.TYPE_2:
            raise ValueError("to_chomsky_normal_form only works for Type-2 grammars")
        
        
        # Step 1: Remove non-terminating symbols
        g1 = self.remove_non_terminating()
        g2 = g1.remove_unreachable()
        g3 = g2.remove_epsilon_productions()
        g4 = g3.remove_unit_productions()
        
        
        new_grammar = Grammar(self.type)
        new_grammar.Sigma = deepcopy(g4.Sigma)
        new_grammar.S = g4.S
        new_grammar.EPSILON = g4.EPSILON
        
        # Counter for new non-terminals
        new_nt_counter_term = 0
        new_nt_counter = 0
        
        # Map from terminal to its non-terminal replacement
        terminal_map = {}
        
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
            
            converted_rhs = []
            for symbol in rhs_tuple:
                if symbol in g4.Sigma:
                    # Replace terminal with non-terminal
                    if symbol not in terminal_map:
                        new_nt_counter_term += 1
                        new_nt = f"X{symbol}"
                        terminal_map[symbol] = new_nt
                        new_grammar.N.add(new_nt)
                        new_grammar.P.add((new_nt, (symbol,)))
                    converted_rhs.append(terminal_map[symbol])
                else:
                    converted_rhs.append(symbol)
            
            # Break down long productions (> 2 symbols)
            if len(converted_rhs) > 2:
                current_lhs = lhs
                for i in range(len(converted_rhs) - 2):
                    new_nt_counter += 1
                    new_nt = f"N{new_nt_counter}"
                    new_grammar.N.add(new_nt)
                    
                    # Create production: current_lhs -> converted_rhs[i] + new_nt
                    new_grammar.P.add((current_lhs, (converted_rhs[i], new_nt)))
                    
                    current_lhs = new_nt
                
                # Final production with last two symbols
                new_grammar.P.add((current_lhs, (converted_rhs[-2], converted_rhs[-1])))
            else:
                # Already in CNF form (2 non-terminals)
                new_grammar.P.add((lhs, tuple(converted_rhs)))
        
        return new_grammar

#dea = Automaton.load_from_file("./dea.txt").pop()
#dea.to_graphviz()

def main():
    """Simple interactive terminal for automaton and grammar operations."""
    automata = {}
    grammars = {}
    
    print("Automaton & Grammar Terminal - Type 'help' for commands\n")
    
    while True:
        try:
            command = input("> ").strip()
            if not command:
                continue
                
            parts = command.split()
            cmd = parts[0].lower()
            
            # Exit
            if cmd in ['exit', 'quit']:
                break
            
            # Help
            elif cmd == 'help':
                print("""
Commands:
  LOADING:
    load <file>              - Load automata/grammars from file
    list                     - List all loaded items
  
  AUTOMATA:
    show <name>              - Show automaton info
    graph <name>             - Visualize automaton
    test <name> <word>       - Test if word is accepted
    minimize <name> [result] - Minimize automaton
    complement <name> [res]  - Complement automaton
    union <n1> <n2> [res]    - Union of automata
    intersect <n1> <n2> [res] - Intersection of automata
  
  GRAMMARS:
    show_grammar <name>      - Show grammar info
    to_cnf <name> [result]   - Convert to Chomsky Normal Form
    to_nea <name> [result]   - Convert Type-3 grammar to NEA
    remove_epsilon <name> [res] - Remove epsilon productions
    remove_unit <name> [res] - Remove unit productions
  
  GENERAL:
    delete <name>            - Delete item
    clear                    - Clear all
    exit                     - Exit
""")
            
            # Load automata/grammars from file
            elif cmd == 'load':
                if len(parts) < 2:
                    print("Usage: load <filename>")
                    continue
                try:
                    loaded_automata, loaded_grammars = load_from_file(parts[1])
                    automata.update(loaded_automata)
                    grammars.update(loaded_grammars)
                    
                    if loaded_automata or loaded_grammars:
                        msg = []
                        if loaded_automata:
                            msg.append(f"{len(loaded_automata)} automata: {', '.join(loaded_automata.keys())}")
                        if loaded_grammars:
                            msg.append(f"{len(loaded_grammars)} grammars: {', '.join(loaded_grammars.keys())}")
                        print(f"Loaded {' and '.join(msg)}")
                    else:
                        print("No items loaded")
                except Exception as e:
                    print(f"Error: {e}")
            
            # List all items
            elif cmd == 'list':
                if automata or grammars:
                    if automata:
                        print("Automata:")
                        for name, aut in sorted(automata.items()):
                            print(f"  {name}: {len(aut.states)} states")
                    if grammars:
                        print("Grammars:")
                        for name, gram in sorted(grammars.items()):
                            print(f"  {name}: {len(gram.N)} non-terminals, {len(gram.P)} productions")
                else:
                    print("Nothing loaded")
            
            # Show automaton info
            elif cmd == 'show':
                if len(parts) < 2:
                    print("Usage: show <name>")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    aut = automata[parts[1]]
                    print(f"\n{parts[1]}:")
                    print(f"  States: {len(aut.states)}")
                    print(f"  Start: {aut.start_state}")
                    print(f"  Accepting: {aut.accepting_states}")
                    print(f"  Transitions: {len(aut.transition_relation)}\n")
            
            # Show grammar info
            elif cmd == 'show_grammar':
                if len(parts) < 2:
                    print("Usage: show_grammar <name>")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    print(grammars[parts[1]])
            
            # Graph automaton
            elif cmd == 'graph':
                if len(parts) < 2:
                    print("Usage: graph <name>")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        automata[parts[1]].to_graphviz(filename=parts[1], view=True)
                        print(f"Created: {parts[1]}.png")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Test word on automaton
            elif cmd == 'test':
                if len(parts) < 3:
                    print("Usage: test <name> <word>")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result = automata[parts[1]].accepts(parts[2])
                        print("ACCEPTED" if result else "REJECTED")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Minimize automaton
            elif cmd == 'minimize':
                if len(parts) < 2:
                    print("Usage: minimize <name> [result]")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result_name = parts[2] if len(parts) > 2 else f"{parts[1]}_min"
                        automata[result_name] = automata[parts[1]].minimize()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Complement automaton
            elif cmd == 'complement':
                if len(parts) < 2:
                    print("Usage: complement <name> [result]")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result_name = parts[2] if len(parts) > 2 else f"{parts[1]}_comp"
                        automata[result_name] = automata[parts[1]].complement()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Union of automata
            elif cmd == 'union':
                if len(parts) < 3:
                    print("Usage: union <n1> <n2> [result]")
                elif parts[1] not in automata or parts[2] not in automata:
                    print("One or both automata not found")
                else:
                    try:
                        result_name = parts[3] if len(parts) > 3 else f"{parts[1]}_union_{parts[2]}"
                        automata[result_name] = automata[parts[1]].union(automata[parts[2]])
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Intersection of automata
            elif cmd == 'intersect':
                if len(parts) < 3:
                    print("Usage: intersect <n1> <n2> [result]")
                elif parts[1] not in automata or parts[2] not in automata:
                    print("One or both automata not found")
                else:
                    try:
                        result_name = parts[3] if len(parts) > 3 else f"{parts[1]}_int_{parts[2]}"
                        automata[result_name] = automata[parts[1]].intersection(automata[parts[2]])
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Convert grammar to CNF
            elif cmd == 'to_cnf':
                if len(parts) < 2:
                    print("Usage: to_cnf <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = parts[2] if len(parts) > 2 else f"{parts[1]}_cnf"
                        grammars[result_name] = grammars[parts[1]].to_chomsky_normal_form()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Convert Type-3 grammar to NEA
            elif cmd == 'to_nea':
                if len(parts) < 2:
                    print("Usage: to_nea <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = parts[2] if len(parts) > 2 else f"{parts[1]}_nea"
                        automata[result_name] = grammars[parts[1]].to_NEA()
                        print(f"Created automaton: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Remove epsilon productions
            elif cmd == 'remove_epsilon':
                if len(parts) < 2:
                    print("Usage: remove_epsilon <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = parts[2] if len(parts) > 2 else f"{parts[1]}_no_eps"
                        grammars[result_name] = grammars[parts[1]].remove_epsilon_productions()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Remove unit productions
            elif cmd == 'remove_unit':
                if len(parts) < 2:
                    print("Usage: remove_unit <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = parts[2] if len(parts) > 2 else f"{parts[1]}_no_unit"
                        grammars[result_name] = grammars[parts[1]].remove_unit_productions()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Delete item
            elif cmd == 'delete':
                if len(parts) < 2:
                    print("Usage: delete <name>")
                else:
                    deleted = False
                    if parts[1] in automata:
                        del automata[parts[1]]
                        deleted = True
                    if parts[1] in grammars:
                        del grammars[parts[1]]
                        deleted = True
                    if deleted:
                        print(f"Deleted: {parts[1]}")
                    else:
                        print(f"Not found: {parts[1]}")
            
            # Clear all
            elif cmd == 'clear':
                automata.clear()
                grammars.clear()
                print("Cleared all")
            
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


def load_from_file(filename: str):
    automata = {}
    grammars = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file contains named items (lines ending with ':' at start of line)
    name_pattern = re.compile(r'^([A-Za-z]\w*):\s*$', re.MULTILINE)
    
    if name_pattern.search(content):
        # Format with names - split into sections
        sections = name_pattern.split(content)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                name = sections[i].strip()
                definition = sections[i + 1].strip()
                
                if not definition:
                    continue
                
                # Determine if it's an automaton or grammar
                is_automaton = detect_automaton(definition)
                
                if is_automaton:
                    # Load as automaton
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
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
                    # Load as grammar
                    try:
                        grammars[name] = Grammar.from_string(definition)
                    except Exception as e:
                        print(f"Warning: Failed to load grammar '{name}': {e}")
    else:
        # Single item without name - detect type and use filename as name
        base_name = os.path.basename(filename).rsplit('.', 1)[0]
        
        is_automaton = detect_automaton(content)
        
        if is_automaton:
            try:
                loaded = Automaton.load_from_file(filename)
                if loaded:
                    for idx, aut in enumerate(loaded):
                        automata[f"{base_name}{idx if idx > 0 else ''}"] = aut
            except Exception as e:
                print(f"Warning: Failed to load automaton: {e}")
        else:
            try:
                grammars[base_name] = Grammar.from_file(filename)
            except Exception as e:
                print(f"Warning: Failed to load grammar: {e}")
    
    return automata, grammars


def detect_automaton(content: str) -> bool:
    """
    Detect if content is an automaton definition or a grammar.
    
    Automaton indicators:
    - Contains 'type:', 'states:', 'alphabet:', 'start:', 'accept:'
    - Contains '->' with state transitions (like 'q0 -> a -> q1')
    
    Grammar indicators:
    - Contains '->' with non-terminal productions (like 'S -> aSb')
    - Contains '::=' (BNF format)
    - Single uppercase letters on left side of productions
    
    Returns:
        bool: True if automaton, False if grammar
    """
    lines = content.strip().split('\n')
    
    # Check for automaton keywords
    automaton_keywords = ['type:', 'states:', 'alphabet:', 'start:', 'accept:']
    has_automaton_keywords = any(
        any(keyword in line.lower() for keyword in automaton_keywords)
        for line in lines
    )
    
    if has_automaton_keywords:
        return True
    
    # Check for BNF format (definitely grammar)
    if '::=' in content:
        return False
    
    # Look at arrow patterns
    # Automaton: 'q0 -> a -> q1' (state -> symbol -> state)
    # Grammar: 'S -> aSb' or 'S -> a' (non-terminal -> symbols)
    
    arrow_lines = [line for line in lines if '->' in line or '→' in line]
    
    if not arrow_lines:
        # No arrows - assume grammar
        return False
    
    # Check pattern: if we see three parts separated by arrows, likely automaton
    automaton_pattern_count = 0
    grammar_pattern_count = 0
    
    for line in arrow_lines:
        parts = re.split(r'\s*->\s*|\s*→\s*', line)
        
        # Automaton typically has 3 parts: state -> symbol -> state
        if len(parts) == 3:
            automaton_pattern_count += 1
        # Grammar typically has 2 parts: non-terminal -> production
        elif len(parts) == 2:
            lhs = parts[0].strip()
            # If LHS is a single uppercase letter, likely grammar
            if len(lhs) == 1 and lhs.isupper():
                grammar_pattern_count += 1
            # If LHS looks like a state name (q0, s1, etc.), likely automaton
            elif lhs.startswith('q') or lhs.startswith('s') or lhs.startswith('S'):
                automaton_pattern_count += 1
    
    # Decide based on which pattern is more common
    return automaton_pattern_count > grammar_pattern_count


if __name__ == "__main__":
    main()



