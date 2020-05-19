/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Models;

import genetics.DataPoint;
import genetics.Networks.Network;
import genetics.Networks.Weights;
import genetics.Validator;
import genetics.Functions.ActivationFunction;
import genetics.Functions.LogisticFunction;
import java.util.HashMap;

/**
 *
 * @author ARZavier
 */
public abstract class GeneticModel {
    Network[] population;
    private int[] shape;
    private double mutation;
    private double crossover;
    ActivationFunction outputFunction;
    
    /**
     * Creates a new evolution-based learning method (GA or DE)
     * @param shape
     * @param popSize
     * @param mutationRate
     * @param crossoverRate
     * @param outputFunc
     */
    
    public GeneticModel(int[] shape, int popSize, double mutationRate, double crossoverRate, ActivationFunction outputFunc) {
        population = new Network[popSize];
        this.shape = shape;
        mutation = mutationRate;
        crossover = crossoverRate;
        outputFunction = outputFunc;
    }
    
    /**
     * Uniform Crossover (Binary Crossover)
     * @param a first parent
     * @param b second parent
     * @return one child
     */
    
    public Network cross(Network a, Network b) {
        Weights childWeights = Weights.cross(a.getWeights(), b.getWeights(), crossover);
        Network n = new Network(childWeights, a.getHiddenFunction(), a.getOutputFunction());
        return n;
    }
    
    /**
     * Mutate the network
     * @param a the network being mutated
     * @return
     */
    
    public Network mutate(Network a){
        a.getWeights().mutate(mutation);
        return a;
    }
    
    public double[] train (DataPoint[] data) {
        double[] metrics = new double[100]; // Just to keep track of the metric at each generation for convergence
        // Initialize the population
        for (int i = 0; i < population.length; i++){population[i] = new Network(shape, new LogisticFunction(), outputFunction);}
        boolean isFinished = false;
        int cnt = 0;
        this.data = data; // So data does not have to be a parameter to getFitness(_)
        System.out.print("Generation: ");
        while(!isFinished) {
            System.out.print(cnt + ", ");
            fillCurrFits(); // Fill in the fitness for the current population so we do not have to recalculate
            // Runs the Model
            population = runGeneration();
            Network net = getMostFit();
            metrics[cnt] = Math.abs(getFitness(net));
            cnt++;
            isFinished = cnt >= 100; // While we have done less than 100 generations
        }
        return metrics;
    }
    
    private DataPoint[] data; // So it does not have to be a parameter to getFitness(_)
    private HashMap<Network, Double> fit;
    /**
     * Stores the fitness of the current population so
     * that we do not have to recalculate every time.
     */
    
    /**
     * Obtains the fitness of a network
     * @param n
     * @return
     */
    public double getFitness(Network n) {
        if (fit.containsKey(n))
            return fit.get(n);
        boolean isClassification = data[0].obtainNumOutput() != 1;
        if (isClassification) {
            // If it is classification, we utilize the accuracy as our fitness measure
            return Validator.accuracy(n, data);
        } else
            return -Validator.squaredError(n, data); // Maximize Negative Squared Error (Minimize Error)
    }
    
    /**
     * Fills the fitness hash map with values from the current population
     */
    
    private void fillCurrFits() {
        fit = new HashMap<>();
        for (Network n : population) {
            double f = getFitness(n);
            fit.put(n, f);
        }
    }
    
    /**
     * Gets the most fit network from the population
     * @return
     */
    
    public Network getMostFit() {
        double maxFitness = Double.NEGATIVE_INFINITY;
        int indx = -1;
        for (int i = 0; i < population.length; i++) {
            if (getFitness(population[i]) > maxFitness) {
                indx = i;
                maxFitness = getFitness(population[i]);
            }
        }
        return population[indx];
    }
    
    /**
     * Runs the selection, crossover, an mutation for an entire generation
     * @return
     */
    public abstract Network[] runGeneration();
}
