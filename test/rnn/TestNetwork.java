package rnn;

import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TestNetwork {

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
