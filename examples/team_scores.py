from kc import dsl
from collections import namedtuple
from functools import reduce

# Example 2: Team Match Scores
# Infer player attributes (binary) from match scores.
# Attributes interact (e.g. Blue doubles strength, Yellow halves it).
# Interaction between teams: "if opponent has blue, subtract 1".

Player = namedtuple("Player", ["name", "has_blue", "has_yellow"])
Team = namedtuple("Team", ["name", "players"])


def make_player(name):
    # Each player has 2 hidden attributes
    return Player(
        name=name,
        has_blue=dsl.flip(0.3, name=f"{name}_blue"),
        has_yellow=dsl.flip(0.3, name=f"{name}_yellow"),
    )


def calculate_strength(team, opponent_team):
    base_strength = 10.0
    total_strength = 0.0

    # Check opponent global properties once
    opponent_has_blue = reduce(
        lambda a, b: a | b, [p.has_blue for p in opponent_team.players]
    )

    for p in team.players:
        s = base_strength

        # "Double if blue"
        s = dsl.if_else(p.has_blue, s * 2.0, s)

        # "Halve if yellow"
        s = dsl.if_else(p.has_yellow, s * 0.5, s)

        # "Subtract 1 if opponent has blue"
        s = dsl.if_else(opponent_has_blue, s - 1.0, s)

        total_strength = total_strength + s

    return total_strength


def infer_team_skills(matches):
    with dsl.Model() as m:
        # We need to implicitly track players created to ensure we reuse same variables for same players?
        # The 'make_player' function creates *new* flips every time it is called.
        # So we must create players *once* outside the match loop/function if they are the same players.
        # But 'matches' usually refer to team names/ids.
        # We'll assume the input 'matches' provides fully instantiated Player objects with DSL variables,
        # OR we maintain a registry.

        # Let's assume input is just data, and we build the model here.
        # For this example, we'll create a fixed set of players.

        pass  # Context active

        # We effectively can't just run 'calculate_strength' on raw data inside 'infer'.
        # We need the 'matches' to pass in the DSL nodes.

    # Re-structure: We create the model environment first.
    pass


def run_example():
    with dsl.Model() as m:
        # Create players
        p1 = make_player("Alice")
        p2 = make_player("Bob")
        p3 = make_player("Charlie")
        p4 = make_player("Dave")

        team_A = Team("A", [p1, p2])
        team_B = Team("B", [p3, p4])

        # Two matches
        matches = [
            (team_A, team_B, 25.0),  # Score for A? Or diff?
            (team_B, team_A, 18.0),  # Score for B
        ]

        for t1, t2, score in matches:
            # Score of t1 against t2
            str_1 = calculate_strength(t1, t2)

            # Score ~ N(strength, 1.0)
            dsl.observe(score == dsl.gaussian(str_1, 1.0))

        # Infer properties of Alice
        # We return the joint distribution of attributes?
        # or just compile the first attribute.

        # compilation target: just one variable to verify.
        ir = m.compile(p1.has_blue)
        return ir


if __name__ == "__main__":
    print("Building team model...")
    ir = run_example()
    print("Compilation successful.")
