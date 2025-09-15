import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # For each varaible, only keep words that are equal to slot length
        for var in self.domains:
            self.domains[var] = {
                w for w in self.domains[var]
                if len(w) == var.length
            }

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps.get((x, y))
        if overlap is None:
            return False

        i, j = overlap
        remove = set()

        # For each candidate of x, check if words in y's domain matches xw at overlap positions
        for xw in self.domains[x]:
            supported = any(
                yw[j] == xw[i]
                for yw in self.domains[y]
            )
            # If no match is found, mark xw for removal
            if not supported:
                remove.add(xw)
        # If found remove them from x's domain and set revised to true
        if remove:
            self.domains[x] -= remove
            revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        if arcs is None:
            queue = [
                (x, y)
                for x in self.crossword.variables
                for y in self.crossword.neighbors(x)
            ]
        else:
            queue = list(arcs)

        # Processes arc in FIFO order
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Checks if all variables are assigned
        return (
            len(assignment) == len(self.crossword.variables)
            and all(v in assignment and assignment[v] is not None
                    for v in self.crossword.variables)
        )

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Assignment is Consistent if
        #  - If all words are correct length
        #  - If all words are distinct
        #  - If all overlapping agree
       
       # All words correct length
        for var, word in assignment.items():
            if len(word) != var.length:
                return False

        # All words distinct
        values = list(assignment.values())
        if len(values) != len(set(values)):
            return False

        # Overlap constraints
        for x in assignment:
            for y in assignment:
                if x == y:
                    continue
                overlap = self.crossword.overlaps.get((x, y))
                if overlap is None:
                    continue
                i, j = overlap
                # If both assigned, letters must match
                if assignment[x][i] != assignment[y][j]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        #
        ordering = []
        for value in self.domains[var]:
            # Count eliminations in unassigned neighbors
            eliminations = 0

            # For unassigned neighbors get overlap positions. For every nhbr word that disagrees increase eliminations count
            for nhbr in self.crossword.neighbors(var):
                if nhbr in assignment:
                    continue
                overlap = self.crossword.overlaps.get((var, nhbr))
                if overlap is None:
                    continue
                i, j = overlap
                for w in self.domains[nhbr]:
                    if w[j] != value[i]:
                        eliminations += 1
            ordering.append((eliminations, value))

        # Sort values by fewest elims 
        ordering.sort(key=lambda t: t[0])
        return [v for eliminations, v in ordering]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        #MRV and degree heuristic
        unassigned = [v for v in self.crossword.variables if v not in assignment]

        def keyFunc(v):
            mrv = len(self.domains[v])
            degree = len(self.crossword.neighbors(v))
            return (mrv, -degree)

        unassigned.sort(key=keyFunc)
        return unassigned[0] if unassigned else None

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        # Recursive backtracking search with MRV and LCV
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        if var is None:
            return None

        for value in self.order_domain_values(var, assignment):
            # Try
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            # Undo
            del assignment[var]

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()