import org.junit.Test;

public class TestNetwork {
    @Test
    public void testNetwork() {
        Network network = new Network();
        network.learn(0.01, 0.01, 0.99, 0.99, 0.0001);

        network.passForward(0.01, 0.01);
        System.out.println(String.format("Output1: %f, output2: %f", network.getOutput(0), network.getOutput(1)));
    }
}
