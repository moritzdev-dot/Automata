from typing_extensions import *

from automaton import Automaton, RegularExpression
from grammar import Grammar
from io_utils import load_from_file


def main():
    """Simple interactive terminal for automaton, grammar, and regex operations."""
    automata: Dict[str, Automaton] = {}
    grammars: Dict[str, Grammar] = {}
    regexes: Dict[str, RegularExpression] = {}

    print("Automaton, Grammar & RegEx Terminal - Type 'help' for commands\n")

    while True:
        try:
            command = input("> ").strip()
            if not command:
                continue

            parts = command.split()
            cmd = parts[0].lower()

            # Exit
            if cmd in ["exit", "quit"]:
                break

            # Help
            elif cmd == "help":
                print(
                    """
Commands:
  LOADING:
    load <file>                  - Load automata/grammars/regexes from file
    list                         - List all loaded items
  
  AUTOMATA OPERATIONS:
    show <name>                  - Show automaton info
    graph <name>                 - Visualize automaton
    test <name> <word>           - Test if word is accepted
    test_pda_final <n> <word>    - Test PDA (type 5) by final state
    test_pda_empty <n> <word>    - Test PDA (type 5) by empty stack
    
    TRANSFORMATIONS:
      to_dea <name> [result]     - Convert NEA/NEA+ε/NEA+words to DEA
      minimize <name> [result]   - Minimize DEA
      complement <name> [res]    - Complement DEA
      
    COMBINATIONS:
      union <n1> <n2> [res]      - Union of two DEAs
      intersect <n1> <n2> [res]  - Intersection of two DEAs
    
    CONVERSIONS:
      to_grammar <name> [result] - Convert automaton to grammar (not implemented)
      to_regex <name> [result]   - Convert automaton to regex
      pda_to_cfg <name> [result] - Convert PDA (type 5) to CFG
  
  GRAMMAR OPERATIONS:
    show_grammar <name>          - Show grammar info
    
    TYPE-2 (CFG) TRANSFORMATIONS:
      remove_non_term <name> [res]  - Remove non-terminating symbols
      remove_unreach <name> [res]   - Remove unreachable symbols
      remove_epsilon <name> [res]   - Remove epsilon productions
      remove_unit <name> [res]      - Remove unit productions
      to_cnf <name> [result]        - Convert to Chomsky Normal Form
      to_pda <name> [result]        - Convert CFG to PDA (empty stack)
      cyk <name> <word>             - CYK membership test (on CNF CFG)
    
    TYPE-3 (REGULAR) CONVERSIONS:
      to_nea <name> [result]        - Convert Type-3 grammar to NEA
  
  REGEX OPERATIONS:
    show_regex <name>            - Show regex pattern
    regex_to_nea <name> [result] - Convert regex to NEA
  
  GENERAL:
    delete <name>                - Delete item
    clear                        - Clear all
    exit                         - Exit
"""
                )

            # Load
            elif cmd == "load":
                if len(parts) < 2:
                    print("Usage: load <filename>")
                    continue
                try:
                    loaded_automata, loaded_grammars, loaded_regexes = load_from_file(
                        parts[1]
                    )
                    automata.update(loaded_automata)
                    grammars.update(loaded_grammars)
                    regexes.update(loaded_regexes)

                    if loaded_automata or loaded_grammars or loaded_regexes:
                        msg = []
                        if loaded_automata:
                            msg.append(
                                f"{len(loaded_automata)} automata: {', '.join(loaded_automata.keys())}"
                            )
                        if loaded_grammars:
                            msg.append(
                                f"{len(loaded_grammars)} grammars: {', '.join(loaded_grammars.keys())}"
                            )
                        if loaded_regexes:
                            msg.append(
                                f"{len(loaded_regexes)} regexes: {', '.join(loaded_regexes.keys())}"
                            )
                        print(f"Loaded {' and '.join(msg)}")
                    else:
                        print("No items loaded")
                except Exception as e:
                    print(f"Error: {e}")

            # List
            elif cmd == "list":
                if automata or grammars or regexes:
                    if automata:
                        print("Automata:")
                        for name, aut in sorted(automata.items()):
                            type_names = {
                                1: "DEA",
                                2: "NEA",
                                3: "NEA+ε",
                                4: "NEA+words",
                                5: "PDA",
                            }
                            print(
                                f"  {name}: {type_names.get(aut.type, 'Unknown')}, {len(aut.states)} states"
                            )
                    if grammars:
                        print("Grammars:")
                        for name, gram in sorted(grammars.items()):
                            print(
                                f"  {name}: {gram.type.name}, {len(gram.N)} non-terminals, {len(gram.P)} productions"
                            )
                    if regexes:
                        print("Regular Expressions:")
                        for name, regex in sorted(regexes.items()):
                            print(f"  {name}: {regex.pattern}")
                else:
                    print("Nothing loaded")

            # Show regex
            elif cmd == "show_regex":
                if len(parts) < 2:
                    print("Usage: show_regex <name>")
                elif parts[1] not in regexes:
                    print(f"Regex not found: {parts[1]}")
                else:
                    print(f"\n{parts[1]}: {regexes[parts[1]].pattern}\n")

            # Convert regex to NEA
            elif cmd == "regex_to_nea":
                if len(parts) < 2:
                    print("Usage: regex_to_nea <name> [result]")
                elif parts[1] not in regexes:
                    print(f"Regex not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_nea"
                        )
                        automata[result_name] = regexes[parts[1]].to_NEA()
                        print(f"Created automaton: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert automaton to regex
            elif cmd == "to_regex":
                if len(parts) < 2:
                    print("Usage: to_regex <name> [result]")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_regex"
                        )
                        regexes[result_name] = automata[parts[1]].to_regex()
                        print(f"Created regex: {result_name}")
                        print(f"Pattern: {regexes[result_name].pattern}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert PDA (type 5) back to CFG (restricted PDAs only)
            elif cmd == "pda_to_cfg":
                if len(parts) < 2:
                    print("Usage: pda_to_cfg <name> [result]")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                elif automata[parts[1]].type != 5:
                    print("pda_to_cfg is only valid for PDA (type 5) automata")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_cfg"
                        )
                        grammars[result_name] = automata[parts[1]].to_CFG()
                        print(f"Created CFG: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Delete item
            elif cmd == "delete":
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
                    if parts[1] in regexes:
                        del regexes[parts[1]]
                        deleted = True
                    if deleted:
                        print(f"Deleted: {parts[1]}")
                    else:
                        print(f"Not found: {parts[1]}")

            # Clear all
            elif cmd == "clear":
                automata.clear()
                grammars.clear()
                regexes.clear()
                print("Cleared all")

            # Show automaton info
            elif cmd == "show":
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
            elif cmd == "show_grammar":
                if len(parts) < 2:
                    print("Usage: show_grammar <name>")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    print(grammars[parts[1]])

            # Graph automaton
            elif cmd == "graph":
                if len(parts) < 2:
                    print("Usage: graph <name>")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        automata[parts[1]].to_graphviz(
                            filename=parts[1], view=True
                        )
                        print(f"Created: {parts[1]}.png")
                    except Exception as e:
                        print(f"Error: {e}")

            # Test word on automaton (NOTE: 'accepts' must exist on Automaton)
            elif cmd == "test":
                if len(parts) < 3:
                    print("Usage: test <name> <word>")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result = automata[parts[1]].accepts(parts[2])  # type: ignore[attr-defined]
                        print("ACCEPTED" if result else "REJECTED")
                    except Exception as e:
                        print(f"Error: {e}")

            # Test PDA by final state
            elif cmd == "test_pda_final":
                if len(parts) < 3:
                    print("Usage: test_pda_final <name> <word>")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                elif automata[parts[1]].type != 5:
                    print("test_pda_final is only valid for PDA (type 5) automata")
                else:
                    try:
                        result = automata[parts[1]].accepts_pda_final_state(parts[2])
                        print("ACCEPTED" if result else "REJECTED")
                    except Exception as e:
                        print(f"Error: {e}")

            # Test PDA by empty stack
            elif cmd == "test_pda_empty":
                if len(parts) < 3:
                    print("Usage: test_pda_empty <name> <word>")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                elif automata[parts[1]].type != 5:
                    print("test_pda_empty is only valid for PDA (type 5) automata")
                else:
                    try:
                        result = automata[parts[1]].accepts_pda_empty_stack(parts[2])
                        print("ACCEPTED" if result else "REJECTED")
                    except Exception as e:
                        print(f"Error: {e}")

            # Minimize automaton
            elif cmd == "minimize":
                if len(parts) < 2:
                    print("Usage: minimize <name> [result]")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_min"
                        )
                        automata[result_name] = automata[parts[1]].minimize()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Complement automaton
            elif cmd == "complement":
                if len(parts) < 2:
                    print("Usage: complement <name> [result]")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_comp"
                        )
                        automata[result_name] = automata[parts[1]].complement()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Union of automata
            elif cmd == "union":
                if len(parts) < 3:
                    print("Usage: union <n1> <n2> [result]")
                elif parts[1] not in automata or parts[2] not in automata:
                    print("One or both automata not found")
                else:
                    try:
                        result_name = (
                            parts[3]
                            if len(parts) > 3
                            else f"{parts[1]}_union_{parts[2]}"
                        )
                        automata[result_name] = automata[parts[1]].union(
                            automata[parts[2]]
                        )
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert NEA/NEA+ε/NEA+words to DEA
            elif cmd == "to_dea":
                if len(parts) < 2:
                    print("Usage: to_dea <name> [result]")
                elif parts[1] not in automata:
                    print(f"Automaton not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_dea"
                        )
                        automata[result_name] = automata[parts[1]].to_DEA()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Intersection of automata
            elif cmd == "intersect":
                if len(parts) < 3:
                    print("Usage: intersect <n1> <n2> [result]")
                elif parts[1] not in automata or parts[2] not in automata:
                    print("One or both automata not found")
                else:
                    try:
                        result_name = (
                            parts[3]
                            if len(parts) > 3
                            else f"{parts[1]}_int_{parts[2]}"
                        )
                        automata[result_name] = automata[parts[1]].intersection(
                            automata[parts[2]]
                        )
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert grammar to CNF
            elif cmd == "to_cnf":
                if len(parts) < 2:
                    print("Usage: to_cnf <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_cnf"
                        )
                        grammars[result_name] = grammars[
                            parts[1]
                        ].to_chomsky_normal_form()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert CFG to PDA (empty stack)
            elif cmd == "to_pda":
                if len(parts) < 2:
                    print("Usage: to_pda <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_pda"
                        )
                        automata[result_name] = grammars[parts[1]].to_PDA()
                        print(f"Created PDA: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # CYK membership test on CFG (assumes CNF)
            elif cmd == "cyk":
                if len(parts) < 3:
                    print("Usage: cyk <grammar_name> <word>")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        accepted = grammars[parts[1]].cyk(parts[2], print_table=True)
                        # cyk already prints table and result; no extra output needed
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert Type-3 grammar to NEA
            elif cmd == "to_nea":
                if len(parts) < 2:
                    print("Usage: to_nea <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_nea"
                        )
                        automata[result_name] = grammars[parts[1]].to_NEA()
                        print(f"Created automaton: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Remove epsilon productions
            elif cmd == "remove_epsilon":
                if len(parts) < 2:
                    print("Usage: remove_epsilon <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_no_eps"
                        )
                        grammars[result_name] = grammars[
                            parts[1]
                        ].remove_epsilon_productions()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Remove unit productions
            elif cmd == "remove_unit":
                if len(parts) < 2:
                    print("Usage: remove_unit <name> [result]")
                elif parts[1] not in grammars:
                    print(f"Grammar not found: {parts[1]}")
                else:
                    try:
                        result_name = (
                            parts[2] if len(parts) > 2 else f"{parts[1]}_no_unit"
                        )
                        grammars[result_name] = grammars[
                            parts[1]
                        ].remove_unit_productions()
                        print(f"Created: {result_name}")
                    except Exception as e:
                        print(f"Error: {e}")

            else:
                print(f"Unknown command: {cmd}")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")

