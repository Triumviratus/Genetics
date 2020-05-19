/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Networks;

import genetics.DataPoint;
import genetics.Functions.ActivationFunction;

/**
 *
 * @author ARZavier
 */
public class Network {
    
    int[] network;
    Weights w;
    ActivationFunction hiddenFunction;
    ActivationFunction outputFunction;
    
    public Network (Weights weights, ActivationFunction hiddenFunction, ActivationFunction outputFunction) {
        w = weights;
        this.hiddenFunction = hiddenFunction;
        this.outputFunction = outputFunction;
        
        network = new int[w.obtainNumberOfLayers() + 1];
        for (int i = 0; i < network.length; i++){network[i] = w.obtainNodesInLayer(i);}
    }
    
    public Network (int inputs, int[] hiddenLayer, int outputs, ActivationFunction hiddenFunc, ActivationFunction outputFunc) {
        network = new int[hiddenLayer.length + 2];
        network[0] = inputs;
        for (int i = 0; i < hiddenLayer.length; i++){network[i + 1] = hiddenLayer[i];}
        network[network.length - 1] = outputs;
        w = new Weights(network); // Initializes Weight Array
        w.randomizeWeights();
        hiddenFunction = hiddenFunc;
        outputFunction = outputFunc;
    }
    
    public Network (int[] shape, ActivationFunction hiddenFunc, ActivationFunction outputFunc) {
        network = shape;
        w = new Weights(network); // Initializes Weight Array
        w.randomizeWeights();
        hiddenFunction = hiddenFunc;
        outputFunction = outputFunc;
    }
    
    double[][] genOutput (DataPoint d){
        double[][] outputs = blankNodes();
        for (int i = 0; i < outputs[0].length; i++){
            // Feed the data point into the network
            outputs[0][i] = d.obtainFieldAt(i);
        }
        for (int layer = 1; layer < outputs.length; layer++) {
            for (int node = 0; node < outputs[layer].length; node++) {
                // Computes the dot product of inputs with weights
                double sum = 0;
                for (int i = 0; i < outputs[layer - 1].length; i++) {
                    /**
                     * It is not the first layer past the inputs, so we need 
                     * to utilize the activation from the previous layer.
                     */
                    if (layer > 1)
                        sum += hiddenFunction.value(outputs[layer - 1][i], new double[]{}) * w.obtainWeight(layer - 1, i, node);
                    else
                        sum += outputs[layer - 1][i] * w.obtainWeight(layer - 1, i, node);
                }
                outputs[layer][node] = sum;
            }
        }
        return outputs; // Returns the output from the network
    }
    
    /**
     * Generates the output from the network given a data point
     * @param d
     * @return
     */
    
    public double[] predict(DataPoint d) {
        
        double[] weightedOut = this.genOutput(d)[network.length-1];
        double[] outs = new double[weightedOut.length];
        for (int i = 0; i < weightedOut.length; i++){outs[i] = outputFunction.value(weightedOut[i], weightedOut);}
        return outs;
    }
    
    /**
     * Creates a double array to store the output values from the network
     * @return
     */
    
    private double[][] blankNodes() {
        double[][] outputs = new double[network.length][];
        for (int i = 0; i < network.length; i++){outputs[i] = new double[network[i]];}
        return outputs;
    }
    
    public int getNumberOfOutputs(){return this.network[this.network.length - 1];}
    public int getNumberOfInputs(){return this.network[0];}
    
    public int getIndexOfOutputLayer(){return this.network.length - 1;}
    public int getNumberOfNodesAtLayer(int layer){return this.network[layer];}
    
    public Weights getWeights(){return w;}
    
    public ActivationFunction getHiddenFunction(){return hiddenFunction;}
    public ActivationFunction getOutputFunction(){return outputFunction;}
    
    public int[] getNetwork(){return network;}
    public void setNetwork(int[] network){this.network = network;}
    public void setWeights(Weights w){this.w = w;}
    
    /**
     * Duplicates the networks to avoid pass by reference issues
     * @return
     */
    
    public Network copy(){
        Network n = new Network(network, hiddenFunction, outputFunction);
        n.w = new Weights(this.w);
        return n;
    }
}
