package neuralnetwork;

import Jama.Matrix;
import org.junit.Test;

import java.io.File;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestNetwork {

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

        assertEquals(o1, network.getOutput(0));
        assertEquals(o2, network.getOutput(1));

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
    public void testPassforward() throws Exception {
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
    public void testLearn1() throws Exception {
        int epochs=0;
        while(epochs++ < 1) {
            Network1 network = new Network1("testLearn1", new int[]{2, 2, 2, 2});
            network.readWeights();

            int i = network.learn(new double[]{0.05, 0.1}, new double[]{0.01, 0.99}, 0.0000001, 20000);
            System.out.println(String.format("%d-%d", epochs, i));
            network.writeWeights();
        }
    }

    @Test
    public void testLearn() throws Exception {
        int epochs=0;
        while(epochs++ < 1) {
            Network network = new Network("testLearn2", new int[]{2, 2, 2, 2});
            network.read();

            int i = network.learn(new double[]{0.05, 0.1}, new double[]{0.01, 0.99}, 0.0000001, 20000);
            System.out.println(String.format("%d-%d", epochs, i));
            network.write();
        }
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
    public void testNetwork2Matrix() throws Exception {
        Network network = new Network("test", new int []{5, 100, 100,  2});
        int iterations = network.learn(new double[]{ 0.99, 0.99, 0.01, 0.01, 0.01 }, new double[] {0.01, 0.99}, 0.000000001);

        network.passForward(new double[]{ 0.99, 0.99, 0.01, 0.01, 0.01 });
        System.out.println(String.format("Iterations: %d, Output1: %f, output2: %f",
                iterations,
                network.getOutput(0),
                network.getOutput(1)));
    }
}
