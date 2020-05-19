/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Functions;

/**
 *
 * @author ARZavier
 */
public abstract class ActivationFunction {
    public ActivationFunction(){}
    
    /**
     * Calculates the value of the activation function
     * @param weightedSum the sum of the output of the upstream node times
     * its weight (i.e., dot product of upstream and weight vectors)
     * @param outputs only relevant for SoftMax
     * @return
     */
    
    public abstract double value(double weightedSum, double[] outputs);
    public abstract double derivative(double weightedSum);
}
