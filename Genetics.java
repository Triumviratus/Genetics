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
import genetics.Functions.ActivationFunction;
import genetics.Functions.Linear;
import genetics.Functions.LogisticFunction;
import genetics.Functions.SoftMax;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

/**
 *
 * @author ARZavier
 */
public class Genetics {
    
    static String[] CATEGORICAL_FEATURES = {"sex", "buying", "maintenance", "doors", "persons", "safety", "lug_boot",
                                            "month", "day", "model"};
    public static String[] classificationFiles = {"abalone", "car", "segmentation"};
    public static String[] regressionFiles = {"wine", "forestfires", "machine"};
    public static String[] allFiles = {"abalone", "car", "segmentation", "forestfires", "wine", "machine"};
    
    private static int[][] allShapes = {{7,9}, {}, {7}, {20,8}, {6,9}, {130,130}};
    private static double[] ffRate = {0.1, 0.1, 0.1, 0.001, 0.01, 0.1};
    
    private static double[][] psoParams = {{4.0, 7.0, 0.4},
                                           {5.0, 6.0, 0.7},
                                           {2.0, 7.0, 0.7},
                                           {2.0, 6.0, 0.7},
                                           {4.0, 7.0, 0.4},
                                           {4.0, 7.0, 0.7}};
    
    private static double[][] deParams = {{0.45, 0.1, 0.1},
                                          {0.6, 0.1, 0.1},
                                          {0.3, 0.9, 0.1},
                                          {0.45, 0.1, 0.2},
                                          {0.45, 0.9, 0.1},
                                          {0.75, 0.5, 0.3}};
    private static double[][] gaParams = {{0.85, 0.5},
                                          {0.55, 0.2},
                                          {0.1, 0.3},
                                          {0.25, 0.40},
                                          {0.7, 0.15},
                                          {0.1, 0.25}};
    static String[] trial = {"car"};
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        for (String path : trial) {
            System.out.println();
            System.out.println();
            System.out.println("Processing: " + path);
            String file = readEntireFile("Data/Assignment3/" + path + "_preprocessed.data"); // reads in the data
            String[] lines = file.split("\r\n");
            DataPoint[] data = new DataPoint[lines.length]; // First line is feature labels
            String[] featureLabels = lines[0].split(",");
            
            // Generate the data points
            for (int i = 1; i < lines.length; i++) {data[i-1] = genDatapoint(lines[i], featureLabels, CATEGORICAL_FEATURES, path);}
            
            data = trim(data);
            DataPoint[][] folds = fold(data); // Splits the data into 10 roughly equal folds
            
            for (int i = 0; i < 1; i++) {
                System.out.println("FOLD: " + i);
                String convHeader = "Generation, multiFF, GA, DE, PSO";
                File convFile = createNewFile("Data/Assignment4/outputs/" + path);
                appendToFile(convHeader, convFile);
                DataPoint[] testData = folds[i];
                DataPoint[] trainData = new DataPoint[0];
                for (int j = 0; j < folds.length; j++) {
                    if (j != i)
                        trainData = concat(folds[j], trainData);
                }
                int inputs = data[0].obtainNumFeatures();
                int outputs = data[0].obtainNumOutput();
                
                int indx = Utilities.indexOf(path, allFiles);
                int[] shape = allShapes[indx]; // Obtains the shape according to the path
                ActivationFunction outputFunc = (trainData[0].obtainNumOutput() == 1) ? new Linear() : new SoftMax();
                
                System.out.println("Inputs " + inputs);
                System.out.println("Outputs " + outputs);
                
                MultiLayerFF ff = new MultiLayerFF(inputs, shape, outputs, new LogisticFunction(), outputFunc);
                double[] ffOuts = ff.train(trainData, ffRate[indx], 100);
                
                GA ga = new GA(getShape(inputs, shape, outputs), 50, gaParams[indx][1], gaParams[indx][0], outputFunc);
                double[] gaOuts = ga.train(trainData);
                
                DE de = new DE(getShape(inputs, shape, outputs), 50, deParams[indx][2], 
                                deParams[indx][1], deParams[indx][0], outputFunc);
                double[] deOuts = de.train(trainData);
                
                PSO pso = new PSO(inputs, shape, outputs, 50, psoParams[indx][0], 
                                    psoParams[indx][1], psoParams[indx][2], outputFunc);
                double[] psoOuts = pso.train(trainData);
                
                // Convergence Outputs
                for (int j = 0; j < gaOuts.length; j++) {
                    String gen = j + ",";
                    gen += ffOuts[j] + ",";
                    gen += gaOuts[j] + ",";
                    gen += deOuts[j] + ",";
                    gen += "" + psoOuts[j];
                    appendToFile(gen, convFile);
                }
            }
        }
    }
    
    private static int[] getShape (int inputs, int[] hiddenLayers, int outputs) {
        int[] shape = new int[hiddenLayers.length + 2];
        shape[0] = inputs;
        for (int i = 0; i < hiddenLayers.length; i++){shape[i+1] = hiddenLayers[i];}
        shape[shape.length-1] = outputs;
        return shape;
    }
    
    /**
     * Creates a data point based on a string input
     * @param featureString String defining the data point
     * @return data point generated
     */
    
    private static DataPoint genDatapoint(String featureString, String[] featureLabels, 
                                          String[] categoricalFeatures, String filename){
        String[] splice = featureString.split(",");
        OneHot hot = new OneHot(filename);
        ArrayList<Double> features = new ArrayList<>();
        
        for (int i = 0; i < splice.length - 1; i++) {
            if (Arrays.asList(categoricalFeatures).contains(featureLabels[i])) {
                // Is Categorical
                String value = splice[i];
                double[] hotVals = hot.obtainOneHot(i, value);
                for (double hv : hotVals){features.add(hv);}
            } else {
                // Is Continuous
                try {
                    features.add(Double.parseDouble(splice[i]));
                } catch (NumberFormatException e){
                    // Is Categorical
                    String value = splice[i];
                    double[] hotVals = hot.obtainOneHot(i, value);
                    for (double hv : hotVals){features.add(hv);}
                }
            }
        }
        
        if (Utilities.contains(filename, classificationFiles)) {
            // Classification
            String classMembership = splice[splice.length - 1];
            double[] classOneHot = hot.obtainOneHot(splice.length - 1, classMembership);
            DataPoint d = new DataPoint(Utilities.convDouble(features), classOneHot);
            return d;
        } else {
            // Regression
            double[] regressionTarget = {Double.parseDouble(splice[splice.length-1])};
            DataPoint d = new DataPoint(Utilities.convDouble(features), regressionTarget);
            return d;
        }
    }
    
    private static String readEntireFile(String filePath){
        // Reads the File
        File file = new File(filePath);
        String retString = "";
        if (file.exists()) {
            try {
                Scanner scan = new Scanner(file);
                scan.useDelimiter("\\Z");
                if (scan.hasNext())
                    retString = scan.next();
                scan.close();
            } catch (FileNotFoundException ignored){
                return "File Not Found For Path: " + file;
            }
        } else
            System.out.println("File Does Not Exist");
        
        return retString;
    }
    
    /**
     * Adds the string to the end of a file
     * @param line string to be added
     * @param file the file to be added to
     */
    
    public static void appendToFile(String line, File file){
        // Adds on to file
        try {
            FileWriter writer = new FileWriter(file, true);
            writer.append(line + '\n');
            writer.close();
        } catch (IOException ignored){}
    }
    
    /**
     * Creates a file if there does not exist one already,
     * then returns the file at the file path.
     * @param filePath file path
     * @return the file (either old or newly created)
     */
    
    public static File createNewFile(String filePath) {
        // Creates a file
        String newPath = filePath;
        File file = new File(newPath + ".csv");
        int i = 2;
        while (file.exists()) {
            newPath = filePath + "-" + i;
            file = new File(newPath + ".csv");
            i += 1;
        } try {
            file.createNewFile();
        } catch (IOException ignored){ignored.printStackTrace();}
        
        return file;
    }
    
    /**
     * Folds the data into 10 relatively equal folds
     * @param points the data to be folded
     * @return a folded list
     */
    
    private static DataPoint[][] fold (DataPoint[] points) {
        
        points = Utilities.scramble(points); // Scrambles the Data
        int folds = 10;
        
        DataPoint[][] data = new DataPoint[folds][points.length];
        int[] counters = new int[folds];
        
        /**
         * So the elements go into the array in order 
         * (i.e., all the null values are at the end).
         */
        Random rand = new Random();
        // Ascertains that all folds have one DataPoint at minimum
        for (int i = 0; i < folds; i++){data[i][counters[i]++] = points[i];}
        
        for (int i = 10; i < points.length; i++) {
            int random = rand.nextInt(folds);
            data[random][counters[random]++] = points[i];
            // Places the points into the folds in order so as to avoid null values
        }
        
        // Eliminates trailing null values
        for (int i = 0; i < data.length; i++){data[i] = trim(data[i]);}
        
        return data;
    }
    
    /**
     * Removes the null values from the end of the array
     * @param points the array
     * @return the values in the same order as they appear in points without trailing null values
     */
    
    private static DataPoint[] trim (DataPoint[] points) {
        int i = 0;
        DataPoint point = points[i++];
        while(point != null){point = points[i++];}
        return (DataPoint[]) Arrays.copyOf(points, i-1);
    }
    
    public static DataPoint[] concat(DataPoint[] a, DataPoint[] b){
        DataPoint[] retMe = new DataPoint[a.length + b.length];
        for (int i = 0; i < a.length; i++){retMe[i] = a[i];}
        for (int i = 0; i < b.length; i++){retMe[i + a.length] = b[i];}
        return retMe;
    }
}