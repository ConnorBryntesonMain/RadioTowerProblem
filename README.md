# Radio Tower Proble

## Made 

This was made in conjunction with Tony Krystofiak physics and computer science double major at Saint John's.

## Overview

The **Radio Tower Problem** addresses the challenge of optimally placing radio towers to ensure comprehensive coverage of a specified area while minimizing the number of towers used.  
This problem is akin to the classic **Set Cover Problem**, a well-known NP-hard problem in computer science.

## Problem Statement

Given a set of potential locations for radio towers and the areas each can cover, determine the minimal subset of towers that collectively cover the entire target area.  
The objective is to achieve full coverage with the least number of towers, optimizing resource utilization.

## Solution Approach

The solution employs a **greedy algorithm** strategy:

1. **Initialization**: Start with an empty set of selected towers.
2. **Iteration**: At each step, select the tower that covers the largest number of uncovered areas.
3. **Update**: Add the selected tower to the solution set and mark its covered areas as covered.
4. **Termination**: Repeat the iteration until all areas are covered.

This approach doesn't always guarantee the optimal solution but provides a solution that's close to optimal in a reasonable computation time, making it practical for large datasets.

## Repository Contents

- **Part3.py**: The main Python script implementing the greedy algorithm for the Radio Tower Problem.
- **RadioTowerCSV/**: Directory containing CSV files with data on tower locations and their coverage areas.
- **RadioTowerImages/**: Directory containing visual representations (e.g., maps or graphs) illustrating tower placements and coverage.

## How to Run

1. **Prerequisites**:
   - Ensure you have Python 3.x installed.
   - Install necessary Python libraries (if any are used in `Part3.py`).

2. **Execution**:
   - Navigate to the repository directory.
   - Run the script:

     ```bash
     python Part3.py
     ```

   - The script will process the data from the CSV files and output the selected tower locations.

## Results
![Screenshot From 2025-04-22 23-25-35](https://github.com/user-attachments/assets/9d25886d-d5fb-43ca-b59f-2828344f457d)

Upon execution, the script outputs a list of selected tower locations that collectively cover the entire target area.  
Visual representations in the `RadioTowerImages` directory provide a graphical overview of the coverage.

## Future Enhancements

- **Algorithm Optimization**: Explore more advanced algorithms (e.g., genetic algorithms, linear programming) to potentially find more optimal solutions.
- **User Interface**: Develop a GUI to allow users to input custom data and visualize results interactively.
- **Real-world Data Integration**: Incorporate real geographical data to apply the solution to actual scenarios. I was able to finsih this up with the help of one Tony Krystofiak, he enabled me to get real world data on what we expanded on.
