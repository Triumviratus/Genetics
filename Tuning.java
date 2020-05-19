/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics;

import genetics.Models.DE;
import genetics.Models.GA;
import genetics.Models.PSO;
import genetics.Networks.MultiLayerFF;
import genetics.Networks.Network;
import genetics.Networks.Particle;
import genetics.Functions.ActivationFunction;
import genetics.Functions.Linear;
import genetics.Functions.LogisticFunction;
import genetics.Functions.SoftMax;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class Tuning {
    
    public static String retMe = "";
    
    /**
     * Tunes a Feed Forward Network
     * This infers the number of inputs and outputs from the dataset
     * @param nodeSearchParams the first parameter is the maximum number of layers, the
     * second is the lower bound of nodes, the third is the upper bound of nodes, and
     * the fourth is the increment between 
     * @param learningRates the learning rates to search
     * @param trainData data to test the model
     * @param testData the tuned model
     * @return  
     */
    
    public static MultiLayerFF tuneFFNetwork(int[] nodeSearchParams, double[] learningRates, 
                                                DataPoint[] trainData, DataPoint[] testData){
        retMe = "";
        int inputs = trainData[0].obtainNumFeatures();
        int outputs = trainData[0].obtainTarget().length;
        
        int[][][] options = buildLayers(nodeSearchParams[0], nodeSearchParams[1], nodeSearchParams[2], nodeSearchParams[3]);
        double best = Double.NEGATIVE_INFINITY;
        MultiLayerFF tunedModel = null;
        
        for (int layer = 0; layer < nodeSearchParams[0]; layer++) {
            for (double rate: learningRates) {
                for (int i = 0; i < options[layer].length; i++) {
                    MultiLayerFF model = new MultiLayerFF(inputs, options[layer][i], outputs, new LogisticFunction(),
                                                          (outputs ==1) ? new Linear() : new SoftMax());
                    model.train(trainData, rate, 100);
                    double value = getMetric(outputs, model, testData);
                    if (value > best){
                        // Set the new model to this one because it is better
                        System.out.println("*BEST*");
                        retMe = "" + rate + "," + Arrays.toString(options[layer][i]) + ",";
                        tunedModel = model;
                        best = value;
                    }
                    System.out.printf("Layers: %d, Learning Rate: %f, Layer Plan: %s \n", 
                            layer, rate, Arrays.toString(options[layer][i]));
                    System.out.println("Accuacy: " + value);
                }
            }
        }
        return tunedModel;
    }
    
    public static DE tuneDE(int[] shape, DataPoint[] trainData, DataPoint[] testData) {
        retMe = "";
        ActivationFunction outputFunc = (trainData[0].obtainNumOutput() ==1) ? new Linear() : new SoftMax();
        int outputs = trainData[0].obtainNumOutput();
        
        double best = Double.NEGATIVE_INFINITY;
        DE tunedDE = null;
        
        for (double beta = 0.3; beta < 1; beta += 0.15) {
            for (double cross = 0.1; cross < 1; cross += 0.2) {
                for (double mut = 0.1; mut < 0.5; mut += 0.1) {
                    DE test = new DE (shape, 50, mut, cross, beta, outputFunc);
                    test.train(testData);
                    Network net = test.getMostFit();
                    double value = getMetric(outputs, net, testData);
                    if (value > best) {
                        best = value;
                        tunedDE = test;
                        System.out.println("\n*BEST*");
                        retMe = beta + ", " + cross + ", " + mut;
                    }
                    System.out.printf("\nBeta %f\tCross %f\tMut %f\n", beta, cross, mut);
                    System.out.println("METRIC: " + value + " Best: " + best);
                }
            }
        }
        return tunedDE;
    }
    
    public static GA tuneGA (int[] shape, DataPoint[] trainData, DataPoint[] testData) {
        retMe = "";
        ActivationFunction outputFunc = (trainData[0].obtainNumOutput() ==1) ? new Linear() : new SoftMax();
        int outputs = trainData[0].obtainNumOutput();
        double best = Double.NEGATIVE_INFINITY;
        GA tunedGA = null;
        
        for (double cross = 0.1; cross < 1; cross += 0.15) {
            for (double mut = 0.1; mut < 0.5; mut += 0.05) {
                GA test = new GA(shape, 50, mut, cross, outputFunc);
                test.train(testData);
                Network net = test.getMostFit();
                double value = getMetric(outputs, net, testData);
                if (value > best) {
                    best = value;
                    tunedGA = test;
                    System.out.println("\n*BEST*");
                    retMe = cross + ", " + mut;
                }
                System.out.printf("\nCross %f\tMut %f\tMut %f\n", cross, mut);
                System.out.println("METRIC: " + value + " Best: " + best);
            }
        }
        return tunedGA;
    }
    
    public static PSO tunePSO (int inputs, int[] shape, int outputs, DataPoint[] trainData, DataPoint[] testData) {
        retMe = "";
        ActivationFunction outputFunc = (trainData[0].obtainNumOutput() ==1) ? new Linear() : new SoftMax();
        double best = Double.NEGATIVE_INFINITY;
        PSO tunedPSO = null;
        
        for (double c1 = 2; c1 < 6; c1 += 1) {
            for (double c2 = 8-c1; c2 < 8; c2 += 1){
                for (double wi = 0.1; wi < 1; wi += 0.3) {
                    PSO test = new PSO (inputs, shape, outputs, 50, c1, c2, wi, outputFunc);
                    test.train(testData);
                    Network net = Particle.gBest;
                    double value = getMetric(outputs, net, testData);
                    if (value > best) {
                        best = value;
                        tunedPSO = test;
                        System.out.println("\n*BEST*");
                        retMe = c1 + ", " + c2 + ", " + wi;
                    }
                    System.out.printf("\nc1 %f\tc2 %f\twi %f\n", c1, c2, wi);
                    System.out.println("METRIC: " + value + " Best: " + best);
                }
            }
        }
        return tunedPSO;
    }
    
    private static double getMetric(int outputs, Network net, DataPoint[] testData){
        double value = 0;
        if (outputs == 1){
            // Regression (Utilize Squared Error)
            value = -Validator.squaredError(net, testData); // Subtract because we seek to maximize negative error
        } else {
            // Classification (Utilize Accuracy)
            value = Validator.accuracy(net, testData);
        }
        return value;
    }
    
    /**
     * Utility function that builds an array of all possible hidden nodes and layer configurations
     * @param layers
     * @param lower
     * @param upper
     * @param inc
     * @return
     */
    
    private static int[][][] buildLayers(int layers, int lower, int upper, int inc) {
        int[] base = new int[(upper - lower)/inc];
        for (int i = 0; i < base.length; i++){base[i] = lower + i * inc;}
        
        int[][][] allOptions = new int[layers + 1][][];
        /**
         * First index is layer number, second is a counter, and
         * the last array stores the configuration.
         */
        allOptions[0] = new int[][] {{}}; // No Hidden Layer
        allOptions[1] = new int[base.length][1]; // One Hidden Layer
        for (int i = 0; i < base.length; i++){allOptions[1][i][0] = base[i];}
        for (int i = 2; i < layers; i++) {
            ArrayList<int[]> layer = new ArrayList<>();
            int[][] prev = allOptions[i - 1];
            for (int j = 0; j < prev.length; j++) {
                for (int k = 0; k < base.length; k++){layer.add(append(base[k], prev[j]));}
            }
            allOptions[i] = flatten(layer);
        }
        return allOptions;
    }
    
    private static int[][] flatten(ArrayList<int[]> l) {
        int[][] retMe = new int[l.size()][];
        for (int i = 0; i < retMe.length; i++){retMe[i] = l.get(i);}
        return retMe;
    }
    
    private static int[] append(int value, int[] arr) {
        int[] retMe = new int[arr.length + 1];
        for (int i = 0; i < arr.length; i++){retMe[i] = arr[i];}
        retMe[retMe.length - 1] = value;
        return retMe;
    }
}
