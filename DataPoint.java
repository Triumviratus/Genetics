/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics;

/**
 *
 * @author ARZavier
 */
public class DataPoint {
    
    private double[] fields;
    private double[] target;
    
    /**
     * Creates a new data point
     * @param fields the one-hot encoded inputs
     * @param target the class or target
     */
    
    DataPoint(double[] fields, double[] target){
        this.fields = fields;
        this.target = target;
    }
    
    public double obtainFieldAt(int index) {return this.fields[index];}
    public double[] obtainTarget(){return this.target;}
    public int obtainNumFeatures(){return fields.length;}
    public int obtainNumOutput(){return target.length;}
    
}
