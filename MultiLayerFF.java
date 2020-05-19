/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Networks;

import genetics.DataPoint;
import genetics.Validator;
import genetics.Functions.ActivationFunction;

/**
 *
 * @author ARZavier
 */
public class MultiLayerFF extends Network {
    
    private Weights momentum;
    public MultiLayerFF (int inputs, int[] hiddenLayer, int outputs, ActivationFunction hiddenFunc, ActivationFunction outputFunc) {
        super(inputs, hiddenLayer, outputs, hiddenFunc, outputFunc);
        momentum = new Weights(w);
        momentum.fill(0.0);
    }
    
    public double obtainFitness(Network n, DataPoint[] data) {
        boolean isClassification = data[0].obtainNumOutput() != 1;
        if(isClassification) {
            return Validator.accuracy(n, data);
        } else
            return Validator.squaredError(n, data); // Maximize negative squared error (i.e., minimize error)
    }
    
    public double[] train (DataPoint[] trainingData, double trainingRate, int maxIterations) {
        int numOfIterations = 0;
        boolean shouldContinue = true;
        double[] metrics = new double[maxIterations];
        while(shouldContinue) {
            // Loop through our training data
            for (DataPoint currPoint : trainingData){backprop(currPoint, trainingRate);}
            metrics[numOfIterations] = obtainFitness(this, trainingData);
            numOfIterations++;
            if (numOfIterations >= maxIterations)
                shouldContinue = false;
        }
        return metrics;
    }
    
    /**
     * @param d the current training data point
     * @return the changes to the weights (deltas)
     */
    
    private void backprop (DataPoint d, double learningRate) {
        double[] target = d.obtainTarget();
        double[][] entireOutput = genOutput(d);
        Weights deltas = new Weights(this.network);
        
        // Gradient descent for the output layers
        int upstreamLayerIndex = getIndexOfOutputLayer() -1;
        // Loop though the output nodes
        for (int outputIndex = 0; outputIndex < this.getNumberOfOutputs(); outputIndex++){
            double error = target[outputIndex] - 
                    outputFunction.value(entireOutput[this.getIndexOfOutputLayer()][outputIndex], 
                            entireOutput[this.getIndexOfOutputLayer()]);
            // loop for the last hidden layer nodes
            for (int upstreamIndex = 0; upstreamIndex < this.getNumberOfNodesAtLayer(upstreamLayerIndex); upstreamIndex++) {
                double derivative = hiddenFunction.derivative(entireOutput[upstreamLayerIndex][upstreamIndex]);
                double change = -error * derivative;
                deltas.setWeight(change, getIndexOfOutputLayer() - 1, upstreamIndex, outputIndex);
            }
        }
        
        // Start at first hidden layer and move back
        for (int layer = getIndexOfOutputLayer() - 1; layer > 0; layer--){
            // Backprop for all other layers
            
            // Loop through the nodes of the hidden layer
            for (int hiddenIndex = 0; hiddenIndex < getNumberOfNodesAtLayer(layer); hiddenIndex++){
                // Loop through upstream nodes
                
                // Generate the sum of the deltas to downstream nodes
                double sum = 0.0;
                for (int downstreamIndex = 0; downstreamIndex < getNumberOfNodesAtLayer(layer + 1); downstreamIndex++) {
                    // Add up changes to all the weights in the downstream layer
                    sum += (deltas.obtainWeight(layer, hiddenIndex, downstreamIndex) * 
                            w.obtainWeight(layer, hiddenIndex, downstreamIndex));
                }
                for (int upstreamIndex = 0; upstreamIndex < getNumberOfNodesAtLayer(layer - 1); upstreamIndex++) {
                    // Set Weight Deltas
                    double change = hiddenFunction.derivative(entireOutput[layer - 1][upstreamIndex] * sum);
                    deltas.setWeight(change, layer - 1, upstreamIndex, hiddenIndex);
                }
            }
        }
        
        // Change the Weights
        for (int layer = 0; layer < network.length - 1; layer++) {
            for (int upstream = 0; upstream < getNumberOfNodesAtLayer(layer); upstream++) {
                for (int downstream = 0; downstream < getNumberOfNodesAtLayer(layer + 1); downstream++) {
                    double dW = -learningRate * deltas.obtainWeight(layer, upstream, downstream) 
                            * hiddenFunction.value(entireOutput[layer][upstream], new double[]{});
                    w.changeWeight(dW, layer, upstream, downstream);
                }
            }
        }
        momentum = deltas;
    }
    
    @Override
    public Weights getWeights(){return w;}
    @Override
    public int[] getNetwork(){return network;}
    @Override
    public void setNetwork(int[] network){this.network = network;}
    @Override
    public void setWeights(Weights w){this.w = w;}
}
