package rnn;

import Jama.Matrix;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.fail;
import static org.junit.Assert.assertTrue;

public class TestNetwork {

    @Test
    public void testReadWrite() throws Exception {
        Network network = new Network("testReadWrite", new int[]{2, 200, 200, 2}, 5);
        network.write();

        double [][] inputs = new double[][] {
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1}
        };

        network.passForward(inputs);

        double o1 = network.getOutput(0);
        double o2 = network.getOutput(1);

        network = new Network("testReadWrite", new int[]{2, 200, 200, 2}, 5);
        network.read();

        network.passForward(inputs);

        assertEquals(o1, network.getOutput(0), 0.0001);
        assertEquals(o2, network.getOutput(1), 0.0001);

        File weights = new File("testReadWrite");
        assertTrue(weights.delete());
    }

    @Test
    public void testGradientChecking() throws Exception {

        double FAULT_TOLERANCE = 0.001;

        int layers [] = {2, 2, 2};

        Network network = new Network("testGradientChecking", layers, 5);
        network.write();

        double epsilon = 0.01;

        double [][] inputs = new double[][] {
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1},
                new double [] {0.05, 0.1}
        };

        double [][] targets = new double[][] {
                new double [] {0.01, 0.99},
                new double [] {0.01, 0.99},
                new double [] {0.01, 0.99},
                new double [] {0.01, 0.99},
                new double [] {0.01, 0.99}
        };

        network.learn(inputs, targets, 0.0000001, 1);

        network.read();

        Matrix gradients, nummericalGradients;

        for(int output=0; output < targets.length; output++) {
            Matrix targetVector = new Matrix(targets[output], targets[output].length);

            for (int layer = layers.length - 1; layer >= 1; layer--) {
                gradients = network.getGradients(output).get(layer - 1);
                nummericalGradients = gradients.copy().times(0);
                Matrix weights = network.getWeights(layer);

                Matrix biasGradients = network.getBiasGradients(output).get(layer - 1);
                Matrix nummericalBiasGradients = biasGradients.copy().times(0);
                Matrix biasWeights = network.getBiasWeights(layer);

                for (int row = 0; row < weights.getRowDimension(); row++) {
                    for (int col = 0; col < weights.getColumnDimension(); col++) {
                        if (col == 0) {
                            double originalBiasWeight = biasWeights.get(row, 0);

                            biasWeights.set(row, 0, originalBiasWeight + epsilon);
                            network.passForward(inputs);

                            double errorPlus = network.error(output, targetVector);

                            biasWeights.set(row, 0, originalBiasWeight - epsilon);
                            network.passForward(inputs);

                            double errorMin = network.error(output, targetVector);

                            biasWeights.set(row, 0, originalBiasWeight);

                            double nummericalGradient = (errorPlus - errorMin) / (2 * epsilon);
                            nummericalBiasGradients.set(row, 0, nummericalGradient);

                            double re = Math.abs(nummericalGradient - biasGradients.get(row, col)) /
                                    (Math.abs(nummericalGradient) + Math.abs(biasGradients.get(row, col)));

                            if (re > FAULT_TOLERANCE) {
                                network.printMatrix(nummericalBiasGradients, "numBiasGrad");
                                network.printMatrix(biasGradients, "biasGrad");

                                fail(String.format("Output: %d, Bias gradient[%d] %.2f %s - %s",
                                        output, layer, re, nummericalGradient, biasGradients.get(row, col)));
                            }
                        }

                        double originalWeight = weights.get(row, col);

                        weights.set(row, col, originalWeight + epsilon);
                        network.passForward(inputs);

                        double errorPlus = network.error(output, targetVector);

                        weights.set(row, col, originalWeight - epsilon);
                        network.passForward(inputs);

                        double errorMin = network.error(output, targetVector);

                        weights.set(row, col, originalWeight);

                        double nummericalGradient = (errorPlus - errorMin) / (2 * epsilon);
                        nummericalGradients.set(row, col, nummericalGradient);

                        double re = Math.abs(nummericalGradient - gradients.get(row, col)) /
                                (Math.abs(nummericalGradient) + Math.abs(gradients.get(row, col)));

                        if (re > FAULT_TOLERANCE) {
                            network.printMatrix(nummericalGradients, "numGrad");
                            network.printMatrix(gradients, "grad");

                            fail(String.format("Output: %d, Gradient[%d] %.2f %s - %s",
                                    output, layer, re, nummericalGradient, gradients.get(row, col)));
                        }
                    }
                }
            }

            if(output == 0) {
                continue;
            }

            Matrix wGradients = network.getWGradients(output);
            Matrix weights = network.getW();
            nummericalGradients = weights.copy().times(0);

            for (int row = 0; row < weights.getRowDimension(); row++) {
                for (int col = 0; col < weights.getColumnDimension(); col++) {
                    double originalWeight = weights.get(row, col);

                    weights.set(row, col, originalWeight + epsilon);
                    network.passForward(inputs);

                    double errorPlus = network.error(output, targetVector);

                    weights.set(row, col, originalWeight - epsilon);
                    network.passForward(inputs);

                    double errorMin = network.error(output, targetVector);

                    weights.set(row, col, originalWeight);

                    double nummericalGradient = (errorPlus - errorMin) / (2 * epsilon);
                    nummericalGradients.set(row, col, nummericalGradient);

                    double re = Math.abs(nummericalGradient - wGradients.get(row, col)) /
                            (Math.abs(nummericalGradient) + Math.abs(wGradients.get(row, col)));

                    if (re > FAULT_TOLERANCE) {

                        network.printMatrix(wGradients, "WGrad");
                        network.printMatrix(nummericalGradients, "numWGrad");

                        fail(String.format("Output: %d, WGradient: %.2f %s - %s",
                                output, re, nummericalGradient, wGradients.get(row, col)));
                    }
                }
            }
        }
    }

    @Test
    public void testLearning() throws Exception {

        Network network = new Network("testRnnLearning", new int []{2, 20, 2}, 5, true);
        network.setLearningRate(0.1);

        double [][] inputs = new double[][] {
            new double[]{0.01, 0.01},
            new double[]{0.99, 0.99},
            new double[]{0.01, 0.01},
            new double[]{0.99, 0.99},
            new double[]{0.01, 0.01}
        };

        double [][] targets = new double[][] {
            new double[]{0, 1},
            new double[]{1, 0},
            new double[]{0, 1},
            new double[]{1, 0},
            new double[]{0, 1}
        };

        int iterations = network.learn(inputs, targets, 0.0001, 0);

        System.out.println("Iterations: " + iterations);

        network.passForward(inputs);

        assertEquals(0, network.getOutputVector(0).get(0, 0), 0.01);
        assertEquals(1, network.getOutputVector(0).get(1, 0), 0.01);

        assertEquals(1, network.getOutputVector(1).get(0, 0), 0.01);
        assertEquals(0, network.getOutputVector(1).get(1, 0), 0.01);

        assertEquals(0, network.getOutputVector(2).get(0, 0), 0.01);
        assertEquals(1, network.getOutputVector(2).get(1, 0), 0.01);

        assertEquals(1, network.getOutputVector(3).get(0, 0), 0.01);
        assertEquals(0, network.getOutputVector(3).get(1, 0), 0.01);

        assertEquals(0, network.getOutputVector(4).get(0, 0), 0.01);
        assertEquals(1, network.getOutputVector(4).get(1, 0), 0.01);
    }
}
