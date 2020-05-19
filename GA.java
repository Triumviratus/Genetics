/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Models;

import genetics.Networks.Network;
import genetics.Functions.ActivationFunction;
import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class GA extends GeneticModel {
    
    /**
     * Creates a new genetic algorithm model to train weights of a neural network
     * @param shape the shape of the neural network being trained
     * @param popSize
     * @param mutationRate
     * @param crossoverRate
     * @param outputFunc
     */
    
    public GA(int[] shape, int popSize, double mutationRate, double crossoverRate, ActivationFunction outputFunc) {
        super(shape, popSize, mutationRate, crossoverRate, outputFunc);
    }
    
    @Override
    public Network[] runGeneration() {
        Network[] nextGen = new Network[population.length]; // Generational Replacement
        for (int i = 0; i < nextGen.length; i++) {
            Network[] parents = select(); // Selects the Parents
            Network child = cross(parents[0], parents[1]); // Crosses the Parents
            child = mutate(child); // Creates mutant
            nextGen[i] = child; // Puts the child into the next generation
        }
        return nextGen;
    }
    
    /**
     * Returns a pair of networks that will serve as the parents
     * @return 
     */
    
    public Network[] select() {
        // Tournment Selection (K = 2)
        Network[] selected = new Network[2]; // To Hold the 2 Parents
        Random rand = new Random();
        for (int i = 0; i < selected.length; i++) {
            // For Each Pair
            Network selected1 = population[rand.nextInt(population.length)];
            // Choose 2 Random Members of the Population
            Network selected2 = population[rand.nextInt(population.length)];
            if (getFitness(selected1) > getFitness(selected2)){
                // Render the More Fit Individual the Parent
                selected[i] = selected1;
            } else
                selected[i] = selected2;
        }
        return selected;
    }
}
