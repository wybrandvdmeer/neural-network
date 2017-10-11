package rnn;

import Jama.Matrix;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.fail;

public class TestNetwork {

    @Test
    public void testGradientChecking() throws Exception {

        double FAULT_TOLERANCE = 0.001;

        int layers [] = {2, 2, 2};

        Network network = new Network("testGradientChecking", layers, 5);
        network.write();

        double epsilon = 0.001;

        double [][] inputs = new double[][] {
                new double[]{0.01, 0.01},
                new double[]{0.99, 0.99},
                new double[]{0.01, 0.01},
                new double[]{0.99, 0.99},
                new double[]{0.01, 0.01}
        };

        double [][] targets = new double[][] {
                new double[]{0.99, 0.99},
                new double[]{0.01, 0.01},
                new double[]{0.99, 0.99},
                new double[]{0.01, 0.01},
                new double[]{0.99, 0.99}
        };

        network.learn(inputs, targets, 0.0000001, 1);

        network.read();

        for(int output=0; output < targets.length; output++) {
            for(int layer=2; layer < layers.length; layer++) {

                Matrix gradients = network.getGradients(output).get(layer - 1);
                Matrix nummericalGradients = gradients.copy();
                Matrix weights = network.getWeights(layer);

                Matrix biasGradients = network.getBiasGradients(output).get(layer - 1);
                Matrix nummericalBiasGradients = biasGradients.copy();
                Matrix biasWeights = network.getBiasWeights(layer);

                Matrix targetVector = new Matrix(targets[output], targets[output].length);

                for(int row=0; row < weights.getRowDimension(); row++) {
                    for(int col=0; col < weights.getColumnDimension(); col++) {
                        double originalWeight = weights.get(row, col);

                        weights.set(row, col, originalWeight + epsilon);
                        network.passForward(inputs);
                        double errorPlus = network.error(output, targetVector);

                        weights.set(row, col, originalWeight - epsilon);
                        network.passForward(inputs);
                        double errorMin = network.error(output, targetVector);

                        weights.set(row, col, originalWeight);

                        double nummericalGradient = (errorPlus - errorMin)/(2 * epsilon);
                        nummericalGradients.set(row, col, nummericalGradient);
                        assertEquals(nummericalGradient, gradients.get(row, col), 0.01);

                        double re = Math.abs(nummericalGradient - gradients.get(row, col))/
                                (Math.abs(nummericalGradient) + Math.abs(gradients.get(row, col)));

                        if(re > FAULT_TOLERANCE) {
                            fail(String.format("Gradient[%d] %.2f %s - %s", layer, re, nummericalGradient, gradients.get(row,0)));
                        }

                        if(col == 0) {
                            double originalBiasWeight = biasWeights.get(row, 0);

                            biasWeights.set(row, 0, originalBiasWeight + epsilon);
                            network.passForward(inputs);
                            errorPlus = network.error(output, targetVector);

                            biasWeights.set(row, 0, originalBiasWeight - epsilon);
                            network.passForward(inputs);
                            errorMin = network.error(output, targetVector);

                            biasWeights.set(row, 0, originalBiasWeight);

                            nummericalGradient = (errorPlus - errorMin)/(2 * epsilon);
                            nummericalBiasGradients.set(row, 0, nummericalGradient);
                            assertEquals(nummericalGradient, biasGradients.get(row, 0), 0.01);

                            re = Math.abs(nummericalGradient - biasGradients.get(row, col))/
                                    (Math.abs(nummericalGradient) + Math.abs(biasGradients.get(row, col)));

                            if(re > FAULT_TOLERANCE) {
                                fail(String.format("Bias gradient[%d] %.2f %s - %s", layer, re, nummericalGradient, biasGradients.get(row,0)));
                            }
                        }
                    }
                }

                network.printMatrix(gradients, "grad");
                network.printMatrix(nummericalGradients, "numGrad");
                network.printMatrix(biasGradients, "biasGrad");
                network.printMatrix(nummericalBiasGradients, "numBiasGrad");
            }
        }
    }

    @Test
    public void testLearning() throws Exception {

        Network network = new Network("test", new int []{2, 2, 2}, 5);

        double [][] inputs = new double[][] {
            new double[]{0.01, 0.01},
            new double[]{0.99, 0.99},
            new double[]{0.01, 0.01},
            new double[]{0.99, 0.99},
            new double[]{0.01, 0.01}
        };

        double [][] targets = new double[][] {
                new double[]{0.99, 0.99},
                new double[]{0.01, 0.01},
                new double[]{0.99, 0.99},
                new double[]{0.01, 0.01},
                new double[]{0.99, 0.99}
        };

        int iterations = network.learn(inputs, targets, 0.0001, 10000);

        System.out.println("Iterations: " + iterations);

        network.passForward(inputs);
        assertEquals(1, network.getOutput(0), 0.0001);
        assertEquals(1, network.getOutput(1), 0.0001);
    }

    @Test
    public void testPassForward() throws Exception {
        Network network = new Network("test", new int []{2, 2, 2}, 2);

        network.getWeights(1).set(0,0,1);
        network.getWeights(1).set(0,1,2);

        network.getWeights(1).set(1,0,3);
        network.getWeights(1).set(1,1,4);

        network.getBiasWeights(1).set(0,0,2);
        network.getBiasWeights(1).set(1,0,2);

        network.getWeights(2).set(0,0,5);
        network.getWeights(2).set(0,1,6);

        network.getWeights(2).set(1,0,7);
        network.getWeights(2).set(1,1,8);

        network.getBiasWeights(2).set(0,0,3);
        network.getBiasWeights(2).set(1,0,3);

        network.getW().set(0,0, 2);
        network.getW().set(0,1, 2);
        network.getW().set(1,0, 2);
        network.getW().set(1,1, 2);

        network.setNoTransfer();

        network.passForward(new double[] {0.5, 1});

        assertEquals(70.5, network.getOutput(0));
        assertEquals(94.5, network.getOutput(1));

        network.nextTimestamp();

        network.passForward(new double[] {0.5, 1});

        assertEquals(334.5, network.getOutput(0));
        assertEquals(454.5, network.getOutput(1));
    }
}
