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
public class Linear extends ActivationFunction {
    
    @Override
    public double value(double weightedSum, double[] outputs){return weightedSum;}
    @Override
    public double derivative(double weightedSum){return 1;}
}
