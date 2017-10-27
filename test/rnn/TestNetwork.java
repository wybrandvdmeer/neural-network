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

        boolean leakyRelu = true;

        Network network = new Network("testGradientChecking", layers, 5, leakyRelu);
        network.write();

        double epsilon = 0.001;

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

        List<Map<Integer, Matrix>> transferDerivativesPerTimeStamp = copyTD(network.getTransferDerivertivesPerTimestamp());

        network.read();

        Matrix gradients, nummericalGradients;

        boolean kinks1, kinks2;

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

                        kinks1 = false;
                        kinks2 = false;

                        if (col == 0) {
                            double originalBiasWeight = biasWeights.get(row, 0);

                            biasWeights.set(row, 0, originalBiasWeight + epsilon);
                            network.passForward(inputs);

                            if(leakyRelu) {
                                kinks1 = checkForKinks(output, transferDerivativesPerTimeStamp, network.getTransferDerivertivesPerTimestamp());
                            }

                            double errorPlus = network.error(output, targetVector);

                            biasWeights.set(row, 0, originalBiasWeight - epsilon);
                            network.passForward(inputs);

                            if(leakyRelu) {
                                kinks2 = checkForKinks(output, transferDerivativesPerTimeStamp, network.getTransferDerivertivesPerTimestamp());
                            }

                            double errorMin = network.error(output, targetVector);

                            biasWeights.set(row, 0, originalBiasWeight);

                            double nummericalGradient = (errorPlus - errorMin) / (2 * epsilon);
                            nummericalBiasGradients.set(row, 0, nummericalGradient);

                            double re = Math.abs(nummericalGradient - biasGradients.get(row, col)) /
                                    (Math.abs(nummericalGradient) + Math.abs(biasGradients.get(row, col)));

                            if (!kinks1 && !kinks2 && re > FAULT_TOLERANCE) {
                                network.printMatrix(nummericalBiasGradients, "numBiasGrad");
                                network.printMatrix(biasGradients, "biasGrad");

                                fail(String.format("Output: %d, Bias gradient[%d] %.2f %s - %s",
                                        output, layer, re, nummericalGradient, biasGradients.get(row, col)));
                            }
                        }

                        kinks1 = false;
                        kinks2 = false;

                        double originalWeight = weights.get(row, col);

                        weights.set(row, col, originalWeight + epsilon);
                        network.passForward(inputs);
                        if(leakyRelu) {
                            kinks1 = checkForKinks(output, transferDerivativesPerTimeStamp, network.getTransferDerivertivesPerTimestamp());
                        }

                        double errorPlus = network.error(output, targetVector);

                        weights.set(row, col, originalWeight - epsilon);
                        network.passForward(inputs);
                        if(leakyRelu) {
                            kinks2 = checkForKinks(output, transferDerivativesPerTimeStamp, network.getTransferDerivertivesPerTimestamp());
                        }

                        double errorMin = network.error(output, targetVector);

                        weights.set(row, col, originalWeight);

                        double nummericalGradient = (errorPlus - errorMin) / (2 * epsilon);
                        nummericalGradients.set(row, col, nummericalGradient);

                        double re = Math.abs(nummericalGradient - gradients.get(row, col)) /
                                (Math.abs(nummericalGradient) + Math.abs(gradients.get(row, col)));

                        if (!kinks1 && !kinks2 && re > FAULT_TOLERANCE) {
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
                    kinks1 = false;
                    kinks2 = false;

                    double originalWeight = weights.get(row, col);

                    weights.set(row, col, originalWeight + epsilon);
                    network.passForward(inputs);
                    if(leakyRelu) {
                        kinks1 = checkForKinks(output, transferDerivativesPerTimeStamp, network.getTransferDerivertivesPerTimestamp());
                    }

                    double errorPlus = network.error(output, targetVector);

                    weights.set(row, col, originalWeight - epsilon);
                    network.passForward(inputs);
                    if(leakyRelu) {
                        kinks2 = checkForKinks(output, transferDerivativesPerTimeStamp, network.getTransferDerivertivesPerTimestamp());
                    }

                    double errorMin = network.error(output, targetVector);

                    weights.set(row, col, originalWeight);

                    double nummericalGradient = (errorPlus - errorMin) / (2 * epsilon);
                    nummericalGradients.set(row, col, nummericalGradient);

                    double re = Math.abs(nummericalGradient - wGradients.get(row, col)) /
                            (Math.abs(nummericalGradient) + Math.abs(wGradients.get(row, col)));

                    if (!kinks1 && !kinks2 && re > FAULT_TOLERANCE) {

                        network.printMatrix(wGradients, "WGrad");
                        network.printMatrix(nummericalGradients, "numWGrad");

                        fail(String.format("Output: %d, WGradient: %.2f %s - %s",
                                output, re, nummericalGradient, wGradients.get(row, col)));
                    }
                }
            }
        }
    }

    private boolean checkForKinks(int maxOutput, List<Map<Integer, Matrix>> td1, List<Map<Integer, Matrix>> td2) {
        for(int output=0; output <= maxOutput; output++) {
            for(int layer=1; layer < td1.get(output).keySet().size(); layer++) {
                Matrix m1 = td1.get(output).get(layer);
                Matrix m2 = td2.get(output).get(layer);
                if(!equals(m1, m2)) {
                    System.out.println("Detected kinks.");
                    return true;
                }
            }
        }
        return false;
    }

    private boolean equals(Matrix m1, Matrix m2) {
        if(m1.getRowDimension() != m2.getRowDimension() || m1.getColumnDimension() != m2.getColumnDimension()) {
            return false;
        }

        for(int row=0; row < m1.getRowDimension(); row++) {
            for(int col=0; col < m1.getColumnDimension(); col++)  {
                if(Math.abs(m1.get(row, col) - m2.get(row, col)) > 0.00001) {
                    return false;
                }
            }
        }
        return true;
    }

    private List<Map<Integer,Matrix>> copyTD(List<Map<Integer, Matrix>> transferDerivertivesPerTimestamp) {
        List<Map<Integer,Matrix>> tPTS = new ArrayList<>();
        for(Map<Integer, Matrix> map : transferDerivertivesPerTimestamp) {
            Map<Integer, Matrix> newMap = new HashMap<>();
            for(int layer : map.keySet()) {
                newMap.put(layer, map.get(layer));
            }
            tPTS.add(newMap);
        }

        return tPTS;
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
            new double[]{0.99, 0.99},
            new double[]{0.01, 0.01},
            new double[]{0.99, 0.99},
            new double[]{0.01, 0.01},
            new double[]{0.99, 0.99}
        };

        int iterations = network.learn(inputs, targets, 0.0001, 0);

        System.out.println("Iterations: " + iterations);

        network.passForward(inputs);

        assertEquals(0.99, network.getOutputVector(0).get(0, 0), 0.01);
        assertEquals(0.99, network.getOutputVector(0).get(1, 0), 0.01);

        assertEquals(0.01, network.getOutputVector(1).get(0, 0), 0.01);
        assertEquals(0.01, network.getOutputVector(1).get(1, 0), 0.01);

        assertEquals(0.99, network.getOutputVector(2).get(0, 0), 0.01);
        assertEquals(0.99, network.getOutputVector(2).get(1, 0), 0.01);

        assertEquals(0.01, network.getOutputVector(3).get(0, 0), 0.01);
        assertEquals(0.01, network.getOutputVector(3).get(1, 0), 0.01);

        assertEquals(0.99, network.getOutputVector(4).get(0, 0), 0.01);
        assertEquals(0.99, network.getOutputVector(4).get(1, 0), 0.01);
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

    private boolean oppositeSign(double d1, double d2) {
        if((d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0)) {
            return true;
        }
        return false;
    }
}
