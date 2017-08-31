package neuralnetwork;

import java.util.Arrays;

public class ScalableNetwork {

    private double learningConstant = 0.5;

    private Neuron [] inputLayer = new Neuron[2];
    private Neuron [] hiddenLayer = new Neuron[2];
    private Neuron [] outputLayer = new Neuron[2];

    private double [][][] weightDerivatives = new double [2][][];
    private double [][] biasWeightDerivatives = new double[2][];

    private static final int HIDDEN_INDEX=0, OUTPUT_INDEX=1;

    public ScalableNetwork(int sizeOfInputLayer, int sizeOfHiddenLayer, int  sizeOfOutputLayer) {

        inputLayer = new Neuron[sizeOfInputLayer];
        hiddenLayer = new Neuron[sizeOfHiddenLayer];
        outputLayer = new Neuron[sizeOfOutputLayer];

        for(int idx=0; idx < inputLayer.length; idx++) {
            inputLayer[idx] = new Neuron(1).isInputNeuron();
        }

        for(int idx=0; idx < hiddenLayer.length; idx++) {
            hiddenLayer[idx] = new Neuron(sizeOfInputLayer);
        }

        for(int idx=0; idx < outputLayer.length; idx++) {
            outputLayer[idx] = new Neuron(sizeOfHiddenLayer);
        }

        weightDerivatives[HIDDEN_INDEX] = new double[sizeOfHiddenLayer][sizeOfInputLayer];
        weightDerivatives[OUTPUT_INDEX] = new double[sizeOfOutputLayer][sizeOfHiddenLayer];

        biasWeightDerivatives[HIDDEN_INDEX] = new double[sizeOfHiddenLayer];
        biasWeightDerivatives[OUTPUT_INDEX] = new double[sizeOfOutputLayer];

        connectLayer(inputLayer, hiddenLayer);
        connectLayer(hiddenLayer, outputLayer);
    }

    public Neuron [][] getLayers() {

        Neuron layers [][] = new Neuron[3][];
        layers[0] = inputLayer;
        layers[1] = hiddenLayer;
        layers[2] = outputLayer;

        return layers;
    }

    private void connectLayer(Neuron [] left, Neuron [] right) {
        for(int idx1=0; idx1 < right.length; idx1++) {
            for(int idx2=0; idx2 < left.length; idx2++) {
                right[idx1].setInput(idx2, left[idx2]);
            }
        }
    }

    public void learn(double [] inputs, double [] targets, double errorLimit) {
        int iterations=0;
        double error;
        while(true) {
            passForward(inputs);

            error = error(targets);

            System.out.println(String.format("It: %d. Error: %f", iterations++, error));

            if(error < errorLimit) {
                break;
            }

            double [] thetas = new double[outputLayer.length];

            for(int idx=0; idx < outputLayer.length; idx++) {
                double output = outputLayer[idx].getOutput();
                thetas[idx] = (output - targets[idx]) * output * (1 - output);
            }

            for(int idx1=0; idx1 < outputLayer.length; idx1++) {
                for(int idx2=0; idx2 < hiddenLayer.length; idx2++) {
                    double pd = thetas[idx1] * hiddenLayer[idx2].getOutput();
                    weightDerivatives[OUTPUT_INDEX][idx1][idx2] = pd;
                }
            }

            for(int idx1=0; idx1 < hiddenLayer.length; idx1++) {
                for (int idx2 = 0; idx2 < inputLayer.length; idx2++) {

                    double hiddenOutput = hiddenLayer[idx1].getOutput();
                    double inputOutput = inputLayer[idx2].getOutput();

                    double pd = hiddenOutput * (1 - hiddenOutput) * inputOutput;

                    double summedTerm = 0;
                    for(int idx3=0; idx3 < thetas.length; idx3++) {
                        summedTerm += (thetas[idx3] * outputLayer[idx3].getWeight(idx1));
                    }

                    pd *= summedTerm;

                    weightDerivatives[HIDDEN_INDEX][idx1][idx2] = pd;
                }
            }

            for(int idx=0; idx < outputLayer.length; idx++) {
                biasWeightDerivatives[OUTPUT_INDEX][idx] = thetas[idx];
            }

            for(int idx1=0; idx1 < hiddenLayer.length; idx1++) {

                double hiddenOutput = hiddenLayer[idx1].getOutput();

                double pd = hiddenOutput * (1 - hiddenOutput);

                double summedTerm=0;
                for(int idx2=0; idx2 < thetas.length; idx2++) {
                    summedTerm += thetas[idx2] * outputLayer[idx2].getWeight(idx1);
                }

                pd *= summedTerm;

                biasWeightDerivatives[HIDDEN_INDEX][idx1] = pd;
            }

            for(int idx1=0; idx1 < outputLayer.length; idx1++) {
                for(int idx2=0; idx2 < hiddenLayer.length; idx2++) {
                    adjustWeight(outputLayer[idx1], idx2, weightDerivatives[OUTPUT_INDEX][idx1][idx2]);
                }
            }

            for(int idx1=0; idx1 < hiddenLayer.length; idx1++) {
                for(int idx2=0; idx2 < inputLayer.length; idx2++) {
                    adjustWeight(hiddenLayer[idx1], idx2, weightDerivatives[HIDDEN_INDEX][idx1][idx2]);
                }
            }

            for(int idx=0; idx < outputLayer.length; idx++) {
                //adjustBiasWeight(outputLayer[idx], biasWeightDerivatives[OUTPUT_INDEX][idx]);
            }

            for(int idx=0; idx < hiddenLayer.length; idx++) {
                adjustBiasWeight(hiddenLayer[idx], biasWeightDerivatives[HIDDEN_INDEX][idx]);
            }
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

    public void passForward(double [] inputs) {
        for(int idx=0; idx < inputs.length; idx++) {
            inputLayer[idx].setInput(inputs[idx]);
        }

        Arrays.stream(inputLayer).forEach(n -> n.fire());
        Arrays.stream(hiddenLayer).forEach(n -> n.fire());
        Arrays.stream(outputLayer).forEach(n -> n.fire());
    }

    private double error(double [] targets) {
        double summedError = 0;

        for(int idx=0; idx < targets.length; idx++) {
            summedError += 0.5 * (targets[idx] - outputLayer[idx].getOutput()) * (targets[idx] - outputLayer[idx].getOutput());
        }

        return summedError;
    }

    public double getOutput(int output) {
        return outputLayer[output].getOutput();
    }
}
