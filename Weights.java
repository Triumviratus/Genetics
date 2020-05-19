/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Networks;

import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class Weights {
    
    protected double[][][] weights;
    public Weights(Weights w){
        // Copies the weights from w into this
        weights = new double[w.weights.length][][];
        for (int layer = 0; layer < w.weights.length; layer++) {
            weights[layer] = new double[w.weights[layer].length][];
            for (int upstream = 0; upstream < w.weights[layer].length; upstream++) {
                weights[layer][upstream] = new double[w.weights[layer][upstream].length];
                for (int downstream = 0; downstream < w.weights[layer][upstream].length; downstream++) {
                    weights[layer][upstream][downstream] = w.weights[layer][upstream][downstream];
                }
            }
        }
    }
    
    public Weights(int[] nodesPerLayer) {
        this.weights = new double[nodesPerLayer.length - 1][][];
        for (int layer = 0; layer < nodesPerLayer.length - 1; layer++){
            this.weights[layer] = new double[nodesPerLayer[layer]][nodesPerLayer[layer + 1]];
        }
    }
    
    public double obtainWeight(int layer, int upstreamIndex, int downstreamIndex) {
        return this.weights[layer][upstreamIndex][downstreamIndex];
    }
    
    public void setWeight(double newWeight, int layer, int upstreamIndex, int downstreamIndex){
        this.weights[layer][upstreamIndex][downstreamIndex] = newWeight;
    }
    
    public void changeWeight(double change, int layer, int upstreamIndex, int downstreamIndex) {
        this.weights[layer][upstreamIndex][downstreamIndex] += change;
    }
    
    public int obtainNumberOfLayers(){return this.weights.length;}
    
    /**
     * Obtains the number of nodes in a given layer
     * @param layer the upstream layer of the edge
     * @return the number of nodes
     */
    
    public int obtainNodesInLayer(int layer){
        if (layer == weights.length){
            // Is Output Layer
            return this.weights[layer-1][0].length;
        }
        return this.weights[layer].length;
    }
    
    /**
     * Performs uniform crossover (binary crossover) of two weights
     * @param a first parent
     * @param b second parent
     * @param crossover the rate at which genes are taken from the second parent
     * 0.5: 50-50 split
     * 1.0: Takes Genes Entirely From Parent b
     * 0.0: Takes Genes Entirely From Parent a
     * @return the new weights object
     */
    
    public static Weights cross (Weights a, Weights b, double crossover) {
        Weights child = new Weights(a);
        Random rand = new Random();
        
        // Goes through each weight in the matrix and flips a coin
        for (int layer = 0; layer < a.weights.length; layer++) {
            for (int upstream = 0; upstream < a.weights[layer].length; upstream++) {
                for (int downstream = 0; downstream < a.weights[layer][upstream].length; downstream++) {
                    if (rand.nextDouble() < crossover){
                        child.weights[layer][upstream][downstream] = b.weights[layer][upstream][downstream];
                    }
                }
            }
        }
        return child;
    }
    
    /**
     * Fills the weights with some constant value
     * @param x
     */
    
    public void fill(double x) {
        for (int layer = 0; layer < weights.length; layer++){
            for (int upstream = 0; upstream < weights[layer].length; upstream++){
                Arrays.fill(weights[layer][upstream], x);
            }
        }
    }
    
    /**
     * Randomize the weights
     */
    
    public void randomizeWeights(){
        Random rand = new Random();
        for (int layer = 0; layer < weights.length; layer++){
            for (int upstream = 0; upstream < weights[layer].length; upstream++) {
                for (int downstream = 0; downstream < weights[layer][upstream].length; downstream++){
                    weights[layer][upstream][downstream] = rand.nextDouble();
                }
            }
        }
    }
    
    /**
     * Mutates the weights
     * @param rate mutation rate (0 --> No Mutations) (1 --> Every Weight Mutates)
     */
    
    public void mutate (double rate) {
        Random rand = new Random();
        double sigma = 10; // Educated Guess
        
        for (int layer = 0; layer < weights.length; layer++) {
            for (int upstream = 0; upstream < weights[layer].length; upstream++) {
                for (int downstream = 0; downstream < weights[layer][upstream].length; downstream++) {
                    if (rand.nextDouble() < rate){
                        double mut;
                        mut = rand.nextGaussian() * sigma;
                        weights[layer][upstream][downstream] += mut;
                    }
                }
            }
        }
    }
    
    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        for (int layer = 0; layer < weights.length; layer++) {
            sb.append ("Layer: ").append(layer).append('\n');
            for (int upstream = 0; upstream < weights[layer].length; upstream++) {
                sb.append("\tUpstream: ").append(upstream).append('\n');
                for (int downstream = 0; downstream < weights[layer][upstream].length; downstream++) {
                    sb.append("\t\tDownstream: ").append(weights[layer][upstream][downstream]).append("\n");
                }
            }
        }
        return sb.toString();
    }
    
    /**
     * Adds the two weight vectors
     * @param a
     * @param b
     * @param scalar a scalar multiple to multiply the sum
     * @return a new weights vector
     */
    
    public static Weights add(Weights a, Weights b, double scalar) {
        Weights s = new Weights(a);
        for (int layer = 0; layer < s.weights.length; layer++) {
            // Goes through each weight and adds them
            for (int upstream = 0; upstream < s.weights[layer][upstream].length; upstream++){
                for (int downstream = 0; downstream < s.weights[layer][upstream].length; downstream++){
                    s.weights[layer][upstream][downstream] = scalar * 
                            (a.weights[layer][upstream][downstream] + b.weights[layer][upstream][downstream]);
                }
            }
        }
        return s;
    }
    
    /**
     * Subtracts b from a (i.e., a - b)
     * scalar * (a-b)
     * @param a
     * @param b
     * @param scalar the constant value to multiply the differences
     * @return
     */
    
    public static Weights sub(Weights a, Weights b, double scalar) {
        Weights s = new Weights(a);
        for (int layer = 0; layer < s.weights.length; layer++) {
            // Goes through each weight and subtracts them
            for (int upstream = 0; upstream < s.weights[layer][upstream].length; upstream++){
                for (int downstream = 0; downstream < s.weights[layer][upstream].length; downstream++){
                    s.weights[layer][upstream][downstream] = scalar * 
                            (a.weights[layer][upstream][downstream] - b.weights[layer][upstream][downstream]);
                }
            }
        }
        return s;
    }
    
    /**
     * Multiplies the weights by some constant value
     * @param a
     * @param scalar
     * @return
     */
    
    public static Weights multiply(Weights a, double scalar) {
        Weights s = new Weights(a);
        for (int layer = 0; layer < s.weights.length; layer++) {
            // Goes through each weight and multiplies them
            for (int upstream = 0; upstream < s.weights[layer][upstream].length; upstream++){
                for (int downstream = 0; downstream < s.weights[layer][upstream].length; downstream++){
                    s.weights[layer][upstream][downstream] = scalar * (a.weights[layer][upstream][downstream]);
                }
            }
        }
        return s;
    }
}
