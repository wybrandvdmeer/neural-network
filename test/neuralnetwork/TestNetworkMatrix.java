package neuralnetwork;

import Jama.Matrix;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TestNetworkMatrix {
    @Test
    public void testPassForward() throws Exception {
        NetworkMatrix network = new NetworkMatrix("test", new int []{2, 2, 2});

        network.getWeights().get(0).set(0,0,1);
        network.getWeights().get(0).set(0,1,2);

        network.getWeights().get(0).set(1,0,3);
        network.getWeights().get(0).set(1,1,4);

        network.printMatrix(network.getWeights().get(0), "weight0");

        network.getWeights().get(1).set(0,0,5);
        network.getWeights().get(1).set(0,1,6);

        network.getWeights().get(1).set(1,0,7);
        network.getWeights().get(1).set(1,1,8);

        network.printMatrix(network.getWeights().get(1), "weight1");

        network.setNoTransfer();

        network.passForward(new double[] {0.5, 1});

        assertEquals(45.5, network.getOutput(0));
        assertEquals(61.5, network.getOutput(1));

    }

    @Test
    public void testLearn() throws Exception {
        NetworkMatrix network = new NetworkMatrix("test", new int[]{2,2,2});
        network.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99}, 0.00001, 1);
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
}
