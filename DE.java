/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Models;

import genetics.Networks.Network;
import genetics.Networks.Weights;
import genetics.Functions.ActivationFunction;
import genetics.Functions.LogisticFunction;

import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class DE extends GeneticModel {
    
    private double beta;
    
    /**
     * Creates a New Differential Evolution Model
     * @param shape the shape of the neural network being trained
     * @param popSize
     * @param mutationRate
     * @param crossoverRate
     * @param beta
     * @param outputFunc
     */
    
    public DE(int[] shape, int popSize, double mutationRate, double crossoverRate, double beta, ActivationFunction outputFunc) {
        super(shape, popSize, mutationRate, crossoverRate, outputFunc);
        this.beta = beta;
    }
    
    @Override
    public Network[] runGeneration(){
        Network[] nextGen = new Network[population.length]; // Generational Replacement
        for (int i = 0; i < nextGen.length; i++){
            // Strategy: DE/rand/1/binary
            Network trial = new Network(trialVector(beta), new LogisticFunction(), outputFunction);
            // Selects the trial vector
            Network child = cross(trial, population[i]);
            child = mutate(child);
            
            if (getFitness(child) > getFitness(population[i])) {
                // If the child is more fit than the parent, put it in the next generation
            } else
                nextGen[i] = population[i]; // Otherwise put the parent back in
        }
        return nextGen;
    }
    
    /**
     * Creates a trial vector from three unique individuals in the population
     * @param beta
     * @return the trial vector
     */
    
    public Weights trialVector(double beta) {
        Random rand = new Random();
        int[] indx = new int[3];
        do {
            indx[0] = rand.nextInt(population.length);
            indx[1] = rand.nextInt(population.length);
            indx[2] = rand.nextInt(population.length);
        } while (indx[0] == indx[1] || indx[0] == indx[2] || indx[1] == indx[2]); // While any of them are the same
        Weights x1 = population[indx[0]].getWeights();
        Weights x2 = population[indx[1]].getWeights();
        Weights x3 = population[indx[2]].getWeights();
        
        Weights differenceVector = Weights.sub(x2, x3, beta); // Calculates the difference vector times beta
        return Weights.add(x1, differenceVector, 1); // Adds it to x1
    }
}
