import java.util.Arrays;

public class Network {

    private double learningConstant = 0.5;

    private Neuron [] inputLayer = new Neuron[2];
    private Neuron [] hiddenLayer = new Neuron[2];
    private Neuron [] outputLayer = new Neuron[2];

    public Network() {
        inputLayer[0] = new Neuron(1).isInputNeuron();
        inputLayer[1] = new Neuron(1).isInputNeuron();

        hiddenLayer[0] = new Neuron(2);
        hiddenLayer[1] = new Neuron(2);

        outputLayer[0] = new Neuron(2);
        outputLayer[1] = new Neuron(2);

        hiddenLayer[0].setInput(0, inputLayer[0]);
        hiddenLayer[0].setInput(1, inputLayer[1]);

        hiddenLayer[1].setInput(0, inputLayer[0]);
        hiddenLayer[1].setInput(1, inputLayer[1]);

        outputLayer[0].setInput(0, hiddenLayer[0]);
        outputLayer[0].setInput(1, hiddenLayer[1]);

        outputLayer[1].setInput(0, hiddenLayer[0]);
        outputLayer[1].setInput(1, hiddenLayer[1]);
    }

    public void learn(double input1, double input2, double target1, double target2, double errorLimit) {
        int iterations=0;
        double error;
        while(true) {
            passForward(input1, input2);

            error = error(target1, target2);

            System.out.println(String.format("It: %d. Error: %f", iterations++, error));

            if(error < errorLimit) {
                break;
            }

            double outputO1 = outputLayer[0].getOutput();
            double outputO2 = outputLayer[1].getOutput();

            double hiddenO1 = hiddenLayer[0].getOutput();
            double hiddenO2 = hiddenLayer[1].getOutput();

            double inputO1 = inputLayer[0].getOutput();
            double inputO2 = inputLayer[1].getOutput();

            double theta1 = (outputO1 - target1) * outputO1 * (1 - outputO1);
            double theta2 = (outputO2 - target2) * outputO2 * (1 - outputO2);

            double w5 = outputLayer[0].getWeight(0);
            double w6 = outputLayer[0].getWeight(1);

            double w7 = outputLayer[1].getWeight(0);
            double w8 = outputLayer[1].getWeight(1);

            double pdW1 = (theta1 * w5 + theta2 * w7) * hiddenO1 * (1 - hiddenO1) * inputO1;
            double pdW2 = (theta1 * w5 + theta2 * w7) * hiddenO1 * (1 - hiddenO1) * inputO2;

            double pdW3 = (theta1 * w6 + theta2 * w8) * hiddenO2 * (1 - hiddenO2) * inputO1;
            double pdW4 = (theta1 * w6 + theta2 * w8) * hiddenO2 * (1 - hiddenO2) * inputO2;

            double pdW5 = theta1 * hiddenO1;
            double pdW6 = theta1 * hiddenO2;

            double pdW7 = theta2 * hiddenO1;
            double pdW8 = theta2 * hiddenO2;

            double pdBiasO1 = theta1;
            double pdBiasO2 = theta2;

            double pdBiasH1 = (theta1 * w5 + theta2 * w7) * hiddenO1 * (1 - hiddenO1);
            double pdBiasH2 = (theta1 * w6 + theta2 * w8) * hiddenO2 * (1 - hiddenO2);


            adjustWeight(hiddenLayer[0], 0, pdW1);
            adjustWeight(hiddenLayer[0], 1, pdW2);

            adjustWeight(hiddenLayer[1], 0, pdW3);
            adjustWeight(hiddenLayer[1], 1, pdW4);

            adjustWeight(outputLayer[0], 0, pdW5);
            adjustWeight(outputLayer[0], 1, pdW6);

            adjustWeight(outputLayer[1], 0, pdW7);
            adjustWeight(outputLayer[1], 1, pdW8);

            adjustBiasWeight(hiddenLayer[0], pdBiasH1);
            adjustBiasWeight(hiddenLayer[1], pdBiasH2);

            adjustBiasWeight(outputLayer[0], pdBiasO1);
            adjustBiasWeight(outputLayer[1], pdBiasO2);
        }
    }

    private void adjustWeight(Neuron neuron, int index, double partialDerivative) {
        double oldWeight = neuron.getWeight(index);
        neuron.setWeight(index, oldWeight - learningConstant * partialDerivative);
    }

    private void adjustBiasWeight(Neuron neuron, double partialDerivative) {
        double oldWeight = neuron.getBiasWeight();
        neuron.setBiasWeight(oldWeight - learningConstant * partialDerivative);
    }

    public void passForward(double input1, double input2) {
        inputLayer[0].setInput(input1);
        inputLayer[1].setInput(input2);

        Arrays.stream(inputLayer).forEach(n -> n.fire());
        Arrays.stream(hiddenLayer).forEach(n -> n.fire());
        Arrays.stream(outputLayer).forEach(n -> n.fire());
    }

    private double error(double target1, double target2) {
        double o1 = outputLayer[0].getOutput();
        double o2 = outputLayer[1].getOutput();
        return 0.5 * (target1 - o1) * (target1 - o1) + 0.5 * (target2 - o2) * (target2 - o2);
    }

    public double getOutput(int output) {
        return outputLayer[output].getOutput();
    }
}
