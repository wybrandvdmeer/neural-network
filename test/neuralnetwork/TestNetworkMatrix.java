package neuralnetwork;

import Jama.Matrix;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TestNetworkMatrix {
    @Test
    public void testPassForward() throws Exception {
        NetworkMatrix network = new NetworkMatrix("test", new int []{2, 2, 2});

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
    public void testPassforward() throws Exception {
        NetworkMatrix network = new NetworkMatrix("test", new int[]{2,2,2});

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
    public void testLearn() throws Exception {
        NetworkMatrix network = new NetworkMatrix("test", new int[]{2,2,2});

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

        assertEquals(0.08216704056423078, network.getGradients(2).get(0, 0)); // First Neuron output layer w11.
        assertEquals(0.08266762784753326, network.getGradients(2).get(0, 1)); // First Neuron output layer w12.

        assertEquals(-0.022602540477475067, network.getGradients(2).get(1, 0)); // First Neuron output layer w21.
        assertEquals(-0.02274024221597822, network.getGradients(2).get(1, 1)); // First Neuron output layer w22.

        assertEquals(0.35891648, network.getWeights(2).get(0,0), 0.00000001);
        assertEquals(0.408666186, network.getWeights(2).get(0,1), 0.00000001);
        assertEquals(0.51130127, network.getWeights(2).get(1,0), 0.00000001);
        assertEquals(0.561370121, network.getWeights(2).get(1,1), 0.00000001);

        assertEquals(0.530750719, network.getBiasWeights(2).get(0,0), 0.00000001);
        assertEquals(0.619049119, network.getBiasWeights(2).get(1,0), 0.00000001);
    }

    public void printMatrix(Matrix matrix, String name) {
        System.out.println("Matrix: " + name);
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                String s = String.format("%.2f", matrix.get(i,j));
                System.out.print(' ');
                System.out.print(s);
            }
            System.out.println();
        }
        System.out.println();
    }

    @Test
    public void testMatrix() {
        Matrix m1 = new Matrix(2,1);
        Matrix m2 = new Matrix(2,1);

        m1.set(0,0,1);
        m1.set(1,0,2);

        m2.set(0,0,3);
        m2.set(1,0,4);


        printMatrix(m1, "m1");
        printMatrix(m2, "m2");
        printMatrix(m2.transpose(), "m2 transpose");

        printMatrix(m1.times(m2.transpose()), "test");
    }
}
