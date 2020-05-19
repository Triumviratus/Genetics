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
public class LogisticFunction extends ActivationFunction {
    @Override
    public double value (double weightedSum, double[] outputs){return (1 / (1 + Math.exp(-weightedSum)));}
    @Override
    public double derivative (double weightedSum){
        double val = value(weightedSum, new double[]{});
        return val * (1-val);
    }
}
