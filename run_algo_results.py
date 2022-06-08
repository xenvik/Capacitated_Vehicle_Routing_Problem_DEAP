from deap_model import *
import argparse
import json

def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_name', type=str, default="./data/json/Input_Data.json", required=False,
                        help="Enter the input Json file name")
    parser.add_argument('--popSize', type=int, default=500, required=False,
                        help="Enter the population size")
    parser.add_argument('--crossProb', type=float, default=0.80, required=False,
                        help="Crossover Probability")
    parser.add_argument('--mutProb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--numGen', type=int, default=200, required=False,
                        help="Number of generations to run")


    args = parser.parse_args()

    # Initializing instance
    nsgaObj = nsga2_algo()

    # Setting internal variables
    nsgaObj.json_instance = input_instance(args.instance_name)
    nsgaObj.pop_size = args.popSize
    nsgaObj.cross_prob = args.crossProb
    nsgaObj.mut_prob = args.mutProb
    nsgaObj.num_gen = args.numGen

    # Running Algorithm
    nsgaObj.run_all()


if __name__ == '__main__':
    main()



if __name__ == "__main__":
    # Plotting Min fitness for each generation
    Fitness_all_plots()

    # Plotting Best Route for last generation
    allpaths, csv_files = results_path()

    # Plotting Route graphs for each vehicle for each result
    for eachpath in allpaths:
        instance = load_csv(eachpath)
        best_route_column = instance['best_individual']
        # get the last row
        best_last_one = best_route_column.iloc[-1]
        csv_title = eachpath.split("/")[-1][:-4]
        best_last_one = json.loads(best_last_one)
        plot_route(best_last_one, csv_title)