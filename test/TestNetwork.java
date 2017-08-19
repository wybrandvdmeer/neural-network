import org.junit.Test;

public class TestNetwork {
    public void testNetwork() {
        Network network = new Network();
        network.learn(0.01, 0.01, 0.99, 0.99, 0.001);

        network.passForward(0.01, 0.01);
        System.out.println(String.format("Output1: %f, output2: %f", network.getOutput(0), network.getOutput(1)));
    }

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
    public void testScalableLengthNetwork() {
        ScalableLengthNetwork scalableLengthNetwork = new ScalableLengthNetwork(new int []{2,2,2});
        scalableLengthNetwork.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99}, 0.001);
    }
}
