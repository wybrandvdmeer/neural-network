package neuralnetwork;

import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TestNetwork {
    @Test
    public void testFirstPass() throws Exception {
        Network network = new Network("test", new int []{2, 2, 2});

        for(int layerIdx = 0; layerIdx < network.getLayers().length; layerIdx++) {
            for(int neuronIdx = 0; neuronIdx < network.getLayers()[layerIdx].length; neuronIdx++) {
                for(int weightIdx = 0; weightIdx < network.getLayers()[layerIdx][neuronIdx].getNoOfWeights(); weightIdx++) {
                    network.getLayers()[layerIdx][neuronIdx].setWeight(weightIdx, layerIdx + 1 + neuronIdx + weightIdx);
                }
                network.getLayers()[layerIdx][neuronIdx].setBiasWeight(1);
            }
        }

        network.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99}, 0.00001, 1);
        double derivatives [][][] = network.getPartialDerivatives();
        assertEquals(0.000967927, derivatives[1][0][0], 0.000000001);
        assertEquals(0.000995353, derivatives[1][0][1], 0.000000001);
        assertEquals(0.000001934, derivatives[1][1][1], 0.000000001);
    }

    @Test
    public void testNetwork() throws Exception {
        Network network = new Network("test", new int []{2, 2, 2});

        Neuron[][] layers = network.getLayers();

        layers[1][0].setBiasWeight(0.35);
        layers[1][1].setBiasWeight(0.35);
        layers[2][0].setBiasWeight(0.6);
        layers[2][1].setBiasWeight(0.6);

        layers[1][0].setWeight(0, 0.15);
        layers[1][0].setWeight(1, 0.2);

        layers[1][1].setWeight(0, 0.25);
        layers[1][1].setWeight(1, 0.3);

        layers[2][0].setWeight(0, 0.4);
        layers[2][0].setWeight(1, 0.45);

        layers[2][1].setWeight(0, 0.5);
        layers[2][1].setWeight(1, 0.55);

        network.learn(new double[]{ 0.05, 0.1 }, new double[] {0.01, 0.99}, 0.00001, 1);

        double [][][] pds = network.getPartialDerivatives();
    }

    @Test
    public void testNetwork2() throws Exception {
        Network network = new Network("test", new int []{5, 100, 100,  2});
        int iterations = network.learn(new double[]{ 0.99, 0.99, 0.01, 0.01, 0.01 }, new double[] {0.01, 0.99}, 0.000000001);

        network.passForward(new double[]{ 0.99, 0.99, 0.01, 0.01, 0.01 });
        System.out.println(String.format("Iterations: %d, Output1: %f, output2: %f",
                iterations,
                network.getOutput(0),
                network.getOutput(1)));
    }
}
