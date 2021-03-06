# Capacitated_Vehicle_Routing_Problem_DEAP
Multiobjective optimisation implementation of capacitated vehicle routing problem with NSGA-II algorithm using deap package.

DEAP model was used for the CVRP MOO using python because DEAP provide very effective and fast optimisation using genetic algorithm.
Previous to applying DEAP model, I have used PULP algorithm for the same CVRP but it is only effective only for clients less than 15.

# Problem Statement
Delivery companies every day need to deliver packages to many different clients. The deliveries are accomplished using an available fleet of vehicles from a central warehouse. The goal of this exercise is to design a route for each vehicle so that all customers are served, and the number of vehicles (objective 1) along with the total traveled distance (objective 2) by all the vehicles are minimized. In addition, the capacity of each vehicle should not be exceeded (constraint 1)

# Installations
1) Python (Packages used DEAP, PANDAS, NUMPY, MATPLOTLIB)
2) Pip
3) Virtual Environment

# Assumptions
1) Due date, service time, ready time are ignored from the dataset as stated in the problem statement
2) There is no cost of extra vehicle. Hence taken = 0.
3) There is no time delay and no time windows for our vehicle at the objective locations
4) Distance between client to client is calculated by Euclidean, as per the task.
5) Vehicle always starts from the depot customer_0 and delivers goods and then comes back to depot again after delivery

# Set up and Running the Project

# 1) Parsing Input

The initial data file which taken out to be in text format is converted in json format for usage in the code.

## Convert .txt to .json format

```bash
python text_to_json.py
```
# 2) Running DEAP model for NSGA-2 Algorithm and getting the visualisation results alongwith

Run the algorithm activating the virtual environment and run this command

```bash
python run_algo_results.py
```

# 3) Running Tests

As stated in the problem statement, inbuilt python module unittest has been used to run all the tests of functions used for implementation of the algorithm.
Run the following command to check the tests

```bash
python -m unittest discover test
```

# Visualisations

We have plotted the results for the most effective routes and subroutes retireved with the implementation of the above algorithm NSGA 2 using DEAP package.
Also results for each genrations have been plotted to see the progress made while increasing the generations in the code. The follwong results are shown for 
### Generation = 200, Population = 500, Crossover Probability = 0.8 and Mutation Probability = 0.02

### RESULTS: Optimised Route Length = 408.1195341859591 and Optimised Number of vehicles = 7

## Most Efficient Vehicle Routing achieved in the last give generation
![Route_Input_Data_Generation200_Population500_Crossover0 8_Mutation0 02](https://user-images.githubusercontent.com/55597813/172602575-2c77ce52-45f5-424e-8751-d2b6b437d51d.png)


## The distance vs generations plot
![Fitness_Input_Data_Generation200_Population500_Crossover0 8_Mutation0 02](https://user-images.githubusercontent.com/55597813/172602627-d347f76a-d7fe-4dfd-a73e-25586af7b3df.png)


# Future Improvement

1) Using different methods for mutations, crossover and selection can result in better performance.
2) Using two MOO together in layers can also be effective in large dataset and optimisation e.g. DEAP and PYMOO.

## License

MIT License
