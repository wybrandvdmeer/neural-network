import org.junit.Test;

public class TestNetwork {
    @Test
    public void testNetwork() {
        Network network = new Network();
        network.learn(0.05, 0.1, 0.01, 0.99, 0.01);
    }
}
