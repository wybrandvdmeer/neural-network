package nn;

import Jama.Matrix;
import org.junit.Test;

import java.io.File;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.fail;
import static org.junit.Assert.assertTrue;

public class TestNetwork {

    @Test
    public void testGradientCheckingRelu() throws Exception {

        double FAULT_TOLERANCE = 0.001;

        int [] layers = new int[] {2, 2, 2, 2};

        Network network = new Network("testGradientChecking", layers, true);
        network.write();

        double epsilon = 0.001;

        double [] input = new double[] {0.05, 0.1};
        double [] target = new double[] {0.01, 0.99};

        Matrix targetVector = new Matrix(target, target.length);
        network.learn(input, target, 0.0000001, 1);

        network.read();

        for(int layer=1; layer < layers.length; layer++) {
            Matrix gradients = network.getGradients(layer);
            Matrix nummericalGradients = gradients.copy();
            Matrix weights = network.getWeights(layer);

            Matrix biasGradients = network.getBiasGradients(layer);
            Matrix nummericalBiasGradients = biasGradients.copy();
            Matrix biasWeights = network.getBiasWeights(layer);

            for(int row=0; row < weights.getRowDimension(); row++) {
                for(int col=0; col < weights.getColumnDimension(); col++) {
                    double originalWeight = weights.get(row, col);

                    weights.set(row, col, originalWeight + epsilon);
                    network.passForward(input);
                    double errorPlus = network.error(targetVector);

                    weights.set(row, col, originalWeight - epsilon);
                    network.passForward(input);
                    double errorMin = network.error(targetVector);

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
                        network.passForward(input);
                        errorPlus = network.error(targetVector);

                        biasWeights.set(row, 0, originalBiasWeight - epsilon);
                        network.passForward(input);
                        errorMin = network.error(targetVector);

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

        File weights = new File("testGradientChecking");
        weights.delete();
    }

    @Test
    public void testGradientChecking() throws Exception {

        double FAULT_TOLERANCE = 0.001;

        int [] layers = new int[] {2, 20, 30, 2};

        Network network = new Network("testGradientChecking", layers);
        network.write();

        double epsilon = 0.001;

        double [] input = new double[] {0.05, 0.1};
        double [] target = new double[] {0.01, 0.99};

        Matrix targetVector = new Matrix(target, target.length);
        network.learn(input, target, 0.0000001, 1);

        network.read();

        for(int layer=1; layer < layers.length; layer++) {
            Matrix gradients = network.getGradients(layer);
            Matrix nummericalGradients = gradients.copy();
            Matrix weights = network.getWeights(layer);

            Matrix biasGradients = network.getBiasGradients(layer);
            Matrix nummericalBiasGradients = biasGradients.copy();
            Matrix biasWeights = network.getBiasWeights(layer);

            for(int row=0; row < weights.getRowDimension(); row++) {
                for(int col=0; col < weights.getColumnDimension(); col++) {
                    double originalWeight = weights.get(row, col);

                    weights.set(row, col, originalWeight + epsilon);
                    network.passForward(input);
                    double errorPlus = network.error(targetVector);

                    weights.set(row, col, originalWeight - epsilon);
                    network.passForward(input);
                    double errorMin = network.error(targetVector);

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
                        network.passForward(input);
                        errorPlus = network.error(targetVector);

                        biasWeights.set(row, 0, originalBiasWeight - epsilon);
                        network.passForward(input);
                        errorMin = network.error(targetVector);

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

        File weights = new File("testGradientChecking");
        weights.delete();
    }

    @Test
    public void testReadWrite() throws Exception {
        Network network = new Network("testReadWrite", new int[]{2,200, 200, 2});
        network.write();

        network.passForward(new double[]{ 0.05, 0.1 });

        double o1 = network.getOutput(0);
        double o2 = network.getOutput(1);

        network = new Network("testReadWrite", new int[]{2,200,200,2});
        network.read();

        network.passForward(new double[]{ 0.05, 0.1 });

        assertEquals(o1, network.getOutput(0), 0.0001);
        assertEquals(o2, network.getOutput(1), 0.0001);

        File weights = new File("testReadWrite");
        assertTrue(weights.delete());
    }

    @Test
    public void testPassForward() throws Exception {
        Network network = new Network("test", new int []{2, 2, 2});

        network.getWeights(1).set(0,0,1);
        network.getWeights(1).set(0,1,2);

        network.getWeights(1).set(1,0,3);
        network.getWeights(1).set(1,1,4);

        network.getBiasWeights(1).set(0,0,2);
        network.getBiasWeights(1).set(1,0,2);

        network.printMatrix(network.getWeights(1), "weight0");

        network.getWeights(2).set(0,0,5);
        network.getWeights(2).set(0,1,6);

        network.getWeights(2).set(1,0,7);
        network.getWeights(2).set(1,1,8);

        network.getBiasWeights(2).set(0,0,3);
        network.getBiasWeights(2).set(1,0,3);

        network.printMatrix(network.getWeights(2), "weight1");

        network.setNoTransfer();

        network.passForward(new double[] {0.5, 1});

        assertEquals(70.5, network.getOutput(0));
        assertEquals(94.5, network.getOutput(1));
    }

    @Test
    public void testPassForward2() throws Exception {
        Network network = new Network("test", new int[]{2,2,2});

        network.getWeights(1).set(0, 0, 0.15);
        network.getWeights(1).set(0, 1, 0.2);
        network.getBiasWeights(1).set(0, 0, 0.35);

        network.getWeights(1).set(1, 0, 0.25);
        network.getWeights(1).set(1, 1, 0.3);
        network.getBiasWeights(1).set(1, 0, 0.35);

        network.getWeights(2).set(0, 0, 0.4);
        network.getWeights(2).set(0, 1, 0.45);
        network.getBiasWeights(2).set(0, 0, 0.6);

        network.getWeights(2).set(1, 0, 0.5);
        network.getWeights(2).set(1, 1, 0.55);
        network.getBiasWeights(2).set(1, 0, 0.6);

        network.passForward(new double[]{ 0.05, 0.1 });

        assertEquals(0.7513650695523157, network.getOutput(0));
        assertEquals(0.7729284653214625, network.getOutput(1));
    }

    @Test
    public void testLearnOnePass() throws Exception {
        Network network = new Network("test", new int[]{2,2,2});

        network.getWeights(1).set(0, 0, 0.15);
        network.getWeights(1).set(0, 1, 0.2);

        network.getWeights(1).set(1, 0, 0.25);
        network.getWeights(1).set(1, 1, 0.3);

        network.getBiasWeights(1).set(0, 0, 0.35);
        network.getBiasWeights(1).set(1, 0, 0.35);

        network.getWeights(2).set(0, 0, 0.4);
        network.getWeights(2).set(0, 1, 0.45);

        network.getWeights(2).set(1, 0, 0.5);
        network.getWeights(2).set(1, 1, 0.55);

        network.getBiasWeights(2).set(0, 0, 0.6);
        network.getBiasWeights(2).set(1, 0, 0.6);

        network.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99}, 0.0000001, 1);
        assertEquals(0.2983711087600027, network.getError());

        // Layer 2.
        assertEquals(0.08216704056423078, network.getGradients(2).get(0, 0)); // w11
        assertEquals(0.08266762784753326, network.getGradients(2).get(0, 1)); // w12.

        assertEquals(-0.022602540477475067, network.getGradients(2).get(1, 0)); // w21.
        assertEquals(-0.02274024221597822, network.getGradients(2).get(1, 1)); // w22.

        assertEquals(0.35891648, network.getWeights(2).get(0,0), 0.00000001);
        assertEquals(0.408666186, network.getWeights(2).get(0,1), 0.00000001);
        assertEquals(0.51130127, network.getWeights(2).get(1,0), 0.00000001);
        assertEquals(0.561370121, network.getWeights(2).get(1,1), 0.00000001);

        assertEquals(0.530750719, network.getBiasWeights(2).get(0,0), 0.00000001);
        assertEquals(0.619049119, network.getBiasWeights(2).get(1,0), 0.00000001);

        // Layer 1.
        assertEquals(0.149780716, network.getWeights(1).get(0, 0), 0.00000001);
        assertEquals(0.199561432, network.getWeights(1).get(0, 1), 0.00000001);

        assertEquals(0.24975114, network.getWeights(1).get(1, 0), 0.00000001);
        assertEquals(0.29950229, network.getWeights(1).get(1, 1), 0.00000001);

        assertEquals(0.345614323, network.getBiasWeights(1).get(0,0), 0.00000001);
        assertEquals(0.345022873, network.getBiasWeights(1).get(1,0), 0.00000001);
    }

    @Test
    public void testPassForwardRelu() throws Exception {
        Network network = new Network("Relu", new int []{2, 2, 2}, true);

        network.getWeights(1).set(0,0, 0.5);
        network.getWeights(1).set(0,1, 0.5);
        network.getBiasWeights(1).set(0,0, 0.1);

        network.getWeights(1).set(1,0, 0.6);
        network.getWeights(1).set(1,1, 0.6);
        network.getBiasWeights(1).set(1,0, 0.1);

        network.getWeights(2).set(0,0, 0.7);
        network.getWeights(2).set(0,1, 0.7);
        network.getBiasWeights(2).set(0,0, 0.2);

        network.getWeights(2).set(1,0, 0.8);
        network.getWeights(2).set(1,1, 0.8);
        network.getBiasWeights(2).set(1,0, 0.2);

        network.passForward(new double[]{0.1, 0.2});

        assertEquals(0.6389938878964051, network.getOutput(0));
        assertEquals(0.6511277386034051, network.getOutput(1));
    }

    @Test
    public void testLearnRelu() throws Exception {
        Network network = new Network("Relu", new int []{2, 2, 2}, true);

        network.getWeights(1).set(0,0, 0.5);
        network.getWeights(1).set(0,1, 0.5);
        network.getBiasWeights(1).set(0,0, 0.1);

        network.getWeights(1).set(1,0, 0.6);
        network.getWeights(1).set(1,1, 0.6);
        network.getBiasWeights(1).set(1,0, 0.1);

        network.getWeights(2).set(0,0, 0.7);
        network.getWeights(2).set(0,1, 0.7);
        network.getBiasWeights(2).set(0,0, 0.2);

        network.getWeights(2).set(1,0, 0.8);
        network.getWeights(2).set(1,1, 0.8);
        network.getBiasWeights(2).set(1,0, 0.2);

        network.learn(new double[]{0.1, 0.2}, new double[] {0.01, 0.99}, 0.0000001, 1);

        assertEquals(0.036274187, network.getGradients(2).get(0, 0), 0.00000001);

        assertEquals(0.003997, network.getGradients(1).get(0, 0), 0.00001);
    }
}
