import sys
import heapq

class Node():
    # state: (int, int)
    # parent: Node
    # action: string
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def actionsMap(actions):
    """Maps 'up', 'down', 'left', 'right' -> 'U/D/L/R'."""
    m = {
        "up": "U",
        "down": "D",
        "left": "L",
        "right": "R"
    }
    return [m[a] for a in actions]
    
def summary(algorithm, solutionTuple, exploredcounter):
    """Prints a summary of the solution."""
    print (f"=== {algorithm} RESULTS ===")
    if solutionTuple is None:
        print("no solution")
        print()
        return
    
    # Cells: tuple of coordinates, these remains unused while printing..
    actions, cells = solutionTuple 
    moveLetter = actionsMap(actions)
    dashedMoves = "-".join(moveLetter)
    pathLength = len(actions)
    pathCost = pathLength

    print(f"Solution (moves): {dashedMoves}")
    print(f"Total path cost: {pathCost}")
    print(f"Path length: {pathLength}")
    print(f"Number of states explored: {exploredcounter}")
    print() 


class Maze():
    # height: Integer
    # width: Integer
    # walls: List of a List of Bools
    # start: (int, int)
    # goal: (int, int)
    # solution: (List strings, List (int, int))
    def __init__(self, filename):

        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()
        self.text = contents.splitlines()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = [] #List of Bools
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None

    # Prints original text maze
    def originalText(self):
        """Print the original maze as found in the input file."""
        print()
        for line in self.text:
            print(line)
        print()

   
    # Prints maze with solution
    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    # (Maze, State) -> List of (Action, (int, int))
    def neighbors(self, state):
        row, col = state
        moves = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in moves:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result


    def IDS(self):
        """Solves maze using Iterative Deepening Search."""

        # Tracks number of states explored
        self.exploredCounter = 0
        self.solution = None
        self.explored = set()

        # Count number of free cells as an upper bound on depth
        freeCells = sum(1 for row in self.walls for col in row if not col)

        startNode = Node(state=self.start, parent=None, action=None)

        # Depth Limited Search
        def DLS(node, limit, pathSet):
            self.exploredCounter += 1
            self.explored.add(node.state)

            # Goal check
            if node.state == self.goal:
                actions, cells = [], []
                current = node
                while current.parent is not None: #backtracks to root node
                    actions.append(current.action)
                    cells.append(current.state)
                    current = current.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return True

            if limit == 0:
                return False

            for action, state in self.neighbors(node.state):
                if state not in pathSet:  # avoid cycles on current path
                    child = Node(state=state, parent=node, action=action)
                    pathSet.add(state)
                    if DLS(child, limit - 1, pathSet):
                        return True
                    pathSet.remove(state)  # backtrack

            return False

        # Keeps looping with increasing depth
        # until solution is found or limit is reached
        for depthLimit in range(freeCells + 1):
            pathSet = {startNode.state}
            if DLS(startNode, depthLimit, pathSet):
                return

        self.solution = None
        return


    def Astar(self):
        """Solves maze using A* search."""
        self.num_explored = 0
        self.solution = None
        self.explored = set()

        start = Node(state=self.start, parent=None, action=None)

        # Priority queue: elements are (priority, count, node)
        frontier = []
        count = 0
        heapq.heappush(frontier, (0, count, start))

        # g(n): cost from start so far
        totalCost = {self.start: 0}

        while frontier:
            _, _, node = heapq.heappop(frontier)
            self.num_explored += 1

            # Goal check
            if node.state == self.goal:
                actions, cells = [], []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            self.explored.add(node.state)

            # Explore neighbors
            for action, state in self.neighbors(node.state):
                newCost = totalCost[node.state] + 1  # all steps cost 1
                if state not in totalCost or newCost < totalCost[state]:
                    totalCost[state] = newCost
                    priority = newCost + manhattan(state, self.goal) #f(n): g(n) + h(n)
                    count += 1  # tie-breaker
                    child = Node(state=state, parent=node, action=action)
                    heapq.heappush(frontier, (priority, count, child))


    # Image Code not important for Traversal
    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

m = Maze(sys.argv[1])

print()
print("Original Maze:")
m.originalText()

# Runs, prints and generates image for IDS search

print("Solving with IDS...")
m.IDS()
if m.solution is None:
    summary("IDS", None, m.exploredCounter)
else:
    summary("IDS", m.solution, m.exploredCounter)
print("Visualizing maze...")
m.print()
m.output_image("mazeIDS.png", show_explored=True)
print("Saved IDS image as mazeIDS.png")
print()

# Runs, prints and generates image for A* search
print("Solving with A*...")
m.Astar()
if m.solution is None:
    summary("A*", None, m.num_explored)
else:
    summary("A*", m.solution, m.num_explored)
print("Visualizing maze...")
m.print()
m.output_image("mazeAstar.png", show_explored=True)
print("Saved A* image as mazeAstar.png")
print()