import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TestNetwork {
    public void testNetwork() {
        Network network = new Network();
        network.learn(0.01, 0.01, 0.99, 0.99, 0.001);

        network.passForward(0.01, 0.01);
        System.out.println(String.format("Output1: %f, output2: %f", network.getOutput(0), network.getOutput(1)));
    }

    @Test
    public void testScalableNetwork() {
        ScalableNetwork network = new ScalableNetwork(2,10,3);
        network.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99, 0.01}, 0.001);

        network.passForward(new double[] { 0.01, 0.01});
        System.out.println(String.format("Output1: %f, output2: %f, output3: %f",
                network.getOutput(0),
                network.getOutput(1),
                network.getOutput(2)));
    }

    @Test
    public void testScalableLengthNetworkFirstPass() {
        ScalableLengthNetwork scalableLengthNetwork = new ScalableLengthNetwork(new int []{2, 2, 2});
        scalableLengthNetwork.initWeights();

        scalableLengthNetwork.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99}, 0.00001, 1);
        double derivatives [][][] = scalableLengthNetwork.getPartialDerivatives();
        assertEquals(0.000967927, derivatives[1][0][0], 0.000000001);
        assertEquals(0.000995353, derivatives[1][0][1], 0.000000001);
        assertEquals(0.000001881, derivatives[1][1][0], 0.000000001);
        assertEquals(0.000001934, derivatives[1][1][1], 0.000000001);
    }

    @Test
    public void testScalableLengthNetwork() {
        ScalableLengthNetwork network = new ScalableLengthNetwork(new int []{2, 100, 100,  2});
        network.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99}, 0.000001);

        network.passForward(new double[] { 0.01, 0.01});
        System.out.println(String.format("Output1: %f, output2: %f",
                network.getOutput(0),
                network.getOutput(1)));
    }
}
