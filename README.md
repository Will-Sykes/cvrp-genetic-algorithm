**Capacitated Vehicle Routing Problem (CVRP) Solver**

**Overview**

This project solves the Capacitated Vehicle Routing Problem (CVRP) using a genetic algorithm. The goal is to determine the optimal routes for a fleet of vehicles to deliver goods to multiple locations while ensuring no vehicle exceeds its capacity and minimizing the total travel cost.

**Features**

Genetic Algorithm: Utilizes a population-based genetic algorithm to generate solutions for the CVRP.

CVRP Datasets: The project works with multiple datasets, each describing cities, their coordinates, demands, and vehicle capacities.

**Files**

Project 3 GACVRP William Sykes.py: Python script implementing the genetic algorithm to solve CVRP instances using the provided datasets.

17city2085min.txt: Distance matrix for a 17-city CVRP problem​(17city2085min).

42citymin699.txt: Distance matrix for a 42-city CVRP problem​(42citymin699).

CVRP22.txt: Data file for a 22-city CVRP problem, including city coordinates, vehicle capacities, and demands​(CVRP22).

CVRP33.txt: Data file for a 33-city CVRP problem​(CVRP33).

CVRP51.txt: Data file for a 51-city CVRP problem​(CVRP51).

**CVRP Data Format**

Each CVRP data file (e.g., CVRP22.txt) contains:

Node Coordinates: The (x, y) coordinates of each city.

Demand Section: The demand of goods at each city.

Depot Section: The starting location for all vehicles.

Capacity: The maximum load each vehicle can carry.

**How to Run**

Ensure you have the necessary Python libraries installed:

bash

pip install numpy

Run the Python script to solve a CVRP instance:

bash

python "Project 3 GACVRP William Sykes.py"

The program will load one of the provided datasets and solve the CVRP using the genetic algorithm.

**Example Datasets**

22-city CVRP (CVRP22.txt):

Number of vehicles: 4

Vehicle capacity: 6000

Optimal value: 375

33-city CVRP (CVRP33.txt):

Number of vehicles: 4

Vehicle capacity: 8000

Optimal value: 835

51-city CVRP (CVRP51.txt):

Number of vehicles: 5

Vehicle capacity: 160

Optimal value: 521

**Future Improvements**

Add visualization of the vehicle routes and their respective solutions.

Fine-tune the genetic algorithm parameters for faster convergence.

Expand the project to handle additional constraints, such as time windows or fuel limits.

**References**

The datasets in this project are from Christophides and Eilon, representing well-known benchmark problems in vehicle routing.
