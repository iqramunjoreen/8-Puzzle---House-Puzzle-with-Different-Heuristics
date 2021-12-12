# 8-Puzzle-&-House-Puzzle-with-Different-Heuristics

This program compared algorithm using different heuristics while solving the 8-puzzle and the house puzzle.
- creates 10 instances of solvable 8-puzzles and 10 instances of House Puzzle
- For each solved problem, records:
   - the total running time in seconds
   - the length (i.e. number of tiles moved) of the solution
   - that total number of nodes that were removed from frontier

The algorithms tested are:
A*-search using the misplaced tile heuristic,
A*-search using the Manhattan distance heuristic (implemented),
A*-search using the max of the misplaced tile heuristic and the Manhattan distance heuristic
