/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Models;

import genetics.DataPoint;
import genetics.Networks.Network;
import genetics.Networks.Particle;

import genetics.Validator;
import genetics.Functions.ActivationFunction;
import genetics.Functions.LogisticFunction;

/**
 *
 * @author ARZavier
 */
public class PSO {
    
    private Particle[] population;
    private int iterations = 100;
    
    /**
     * Creates a PSO model to train a neural network
     * @param inputs the number of inputs to the model
     * @param shape the shape of the hidden layers
     * @param outputs the outputs from the model
     * @param pop_size
     * @param c1
     * @param c2
     * @param wi
     * @param function
     */
    
    public PSO(int inputs, int[] shape, int outputs, int pop_size, double c1, double c2, double wi, ActivationFunction function) {
        population = new Particle[pop_size];
        for (int i = 0; i < pop_size; i++){
            population[i] = new Particle(inputs, shape, outputs, c1, c2, wi, new LogisticFunction(), function);
        }
        Particle.gBest = population[0].copy();
    }
    
    /**
     * Trains the PSO
     * @param data
     * @return the metrics for each generation
     */
    
    public double[] train(DataPoint[] data){
        System.out.print("It: ");
        double[] metrics = new double[iterations]; // Just to keep track of the metric at each generation for convergence
        for (int i = 0; i < iterations; i++){
            System.out.print(i + ", ");
            for (Particle p : population) {
                updatePopPosition(p, data);
                // Update the position and velocity of each particle
            }
            metrics[i] = Math.abs(fitness);
        }
        return metrics;
    }
    
    /**
     * Updates the particle velocity and position and also
     * checks if its better than pBest and gBest.
     * @param p
     * @param data
     */
    public void updatePopPosition(Particle p, DataPoint[] data) {
        p.updateVelocity();
        p.updateState();
        if (getFitness(p, data) > getFitness(p.getpBest(), data)){p.setpBest(p);}
        if (getFitness(p, data) > getFitness(Particle.gBest, data)){
            p.setgBest(p);
            fitness = getFitness(p, data);
            lastGBest = p;
        }
    }
    
    public Network lastGBest; // Stores the last gBest so we avoid recalculating its fitness
    private double fitness; // The fitness of the last gBest we found
    
    public double getFitness (Network n, DataPoint[] data){
        if (n == lastGBest) {
            // Checks that we are not calculating the gBest
            return fitness;
        }
        boolean isClassification = data[0].obtainNumOutput() != 1;
        if (isClassification)
            return Validator.accuracy(n, data); // Utilize Accuracy for Classification Metric
        else
            return -Validator.squaredError(n, data); // Maximize Negative Squared Error (Minimize Error)
    }
}
