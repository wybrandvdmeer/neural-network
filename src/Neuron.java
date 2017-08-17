import java.util.Random;

public class Neuron {

    private double input;
    private double output;

    private double biasWeight;

    private boolean isInputType = false;

    private Random random = new Random();
    private Neuron [] inputs;

    private double [] weights;

    public Neuron(int noOfInputs) {
        weights = new double[noOfInputs];
        inputs = new Neuron[noOfInputs];
        for(int idx=0;idx < noOfInputs; idx++) {
            weights[idx] = -1 + 2 * random.nextDouble();
        }

        biasWeight = -1 + 2 * random.nextDouble();
    }

    private double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    public void setInput(int index, Neuron input) {
        inputs[index] = input;
    }

    public Neuron isInputNeuron() {
        isInputType = true;
        return this;
    }

    public double getOutput() {
        return output;
    }

    public void setInput(double input) {
        this.input = input;
    }

    public void fire() {
        if(isInputType) {
            output = input;
        } else {
            double summedInput = 0;
            for(int idx=0; idx < inputs.length; idx++) {
                summedInput += weights[idx] * inputs[idx].getOutput();
            }

            summedInput += biasWeight;

            output = sigmoid(summedInput);
        }
    }

    public double getWeight(int index) {
        return weights[index];
    }

    public void setWeight(int index, double weight) {
        weights[index] = weight;
    }

    public double getBiasWeight() {
        return biasWeight;
    }

    public void setBiasWeight(double weight) {
        this.biasWeight = weight;
    }
}
