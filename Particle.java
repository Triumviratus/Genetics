/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetics.Networks;

import genetics.Functions.ActivationFunction;
import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class Particle extends Network {
    
    public static Network gBest;
    /**
     * The best state that the model has found, which
     * is static so that all particles see the same gBest.
     */
    private Network pBest; // Personal Best of the Particle
    private Weights velocity;
    private double c2 = 1.8;
    private double c1 = 2.3;
    private double wi = 0.8;
    private int t;
    private Random rand = new Random();
    
    /**
     * Creates a particle that is utilized for PSO
     * @param inputs
     * @param hiddenLayers
     * @param outputs
     * @param c1
     * @param c2
     * @param wi
     * @param function
     * @param outputFunc
     */
    
    public Particle(int inputs, int[] hiddenLayers, int outputs, double c1, double c2, double wi, ActivationFunction function,
                    ActivationFunction outputFunc){
        super(inputs, hiddenLayers, outputs, function, outputFunc);
        this.c1 = c1;
        this.c2 = c2;
        this.wi = wi;
        t = 0;
        pBest = this.copy();
        velocity = new Weights(this.w);
        velocity.fill(0); // Start Them Not Moving
    }
    
    /**
     * Moves the particle based upon its velocity
     */
    public void updateState(){this.setWeights(Weights.add(this.getWeights(), this.velocity, 1));}
    
    /**
     * Updates the velocity based on the social and cognitive aspects of PSO
     */
    public void updateVelocity(){
        double r1 = rand.nextDouble();
        double r2 = rand.nextDouble();
        Weights cognitive = Weights.sub(pBest.getWeights(), this.getWeights(), c1 * r1);
        Weights social = Weights.sub(gBest.getWeights(), this.getWeights(), c2 * r2);
        Weights add1 = Weights.add(cognitive, social, 1);
        Weights inertia = Weights.multiply(velocity, wi);
        this.velocity = Weights.add(inertia, add1, 1);
        t++;
    }
    
    public Network getpBest(){return pBest;}
    public void setgBest(Network best){gBest = best;}
    public void setpBest(Network best){this.pBest = best;}
}
