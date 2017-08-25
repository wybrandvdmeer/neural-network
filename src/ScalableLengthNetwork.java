import java.util.Arrays;

public class ScalableLengthNetwork {

    private double learningConstant = 0.5;

    private Neuron [][] layers; // 0 -> input layer.

    private double [][][] weightDerivatives;

    public ScalableLengthNetwork(int [] layerSizes) {
        layers = new Neuron[layerSizes.length][];
        weightDerivatives = new double[layerSizes.length - 1][][];

        for(int idx1=0; idx1 < layerSizes.length; idx1++) {
            layers[idx1] = new Neuron[layerSizes[idx1]];
            for(int idx2=0; idx2 < layerSizes[idx1]; idx2++) {
                if(idx1 ==0) {
                    layers[idx1][idx2] = new Neuron(1).isInputNeuron();
                } else {
                    layers[idx1][idx2] = new Neuron(layers[idx1 - 1].length);
                }
            }
        }

        for(int idx=0; idx < layerSizes.length - 1; idx++) {
            connectLayer(layers[idx], layers[idx + 1]);
        }

        for(int idx1=1; idx1 < layerSizes.length; idx1++) {
            weightDerivatives[idx1 - 1] = new double[layerSizes[idx1]][layerSizes[idx1-1]];
        }
    }

    public void initWeights() {
        for(int layerIdx=0; layerIdx < layers.length; layerIdx++) {
            for(int neuronIdx=0; neuronIdx < layers[layerIdx].length; neuronIdx++) {
                for(int weightIdx=0; weightIdx < layers[layerIdx][neuronIdx].getNoOfWeights(); weightIdx++) {
                    layers[layerIdx][neuronIdx].setWeight(weightIdx, layerIdx + 1 + neuronIdx + weightIdx);
                }
                layers[layerIdx][neuronIdx].setBiasWeight(1);
            }
        }
    }

    public double [][][] getPartialDerivatives() {
        return weightDerivatives;
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
        passForward(inputs);

        error = error(targets);

        double [] thetas = calculateThetas(targets);

        // Propagating backwards.
        for(int layerIdx=layers.length - 1; layerIdx > 0; layerIdx--) {
            for(int neuronIdx=0; neuronIdx < layers[layerIdx].length; neuronIdx++) {
                Neuron neuron = layers[layerIdx][neuronIdx];
                for (int weightIdx = 0; weightIdx < layers[layerIdx].length; weightIdx++) {
                    double pd = calculateSigmoidDerivativeTimesOutputPrevLayer(neuronIdx, layerIdx, weightIdx);
                    if(layerIdx == layers.length - 1) {
                        pd *= (neuron.getOutput() - targets[neuronIdx]);
                    }

                    weightDerivatives[layerIdx - 1][neuronIdx][weightIdx] = pd;
                }
            }
        }

        System.out.println(String.format("It: %d. Error: %f", iterations++, error));
    }

    /*
    theta1 = (O1out - T1) * O11out (1 - O1out)
    theta1 = (O1out - T1) * sigmoid-deriv(O1)
    */
    private double [] calculateThetas(double [] targets) {
        int outputLayerIdx = layers.length - 1;
        int widthOuputLayer = layers[outputLayerIdx].length;
        double [] thetas = new double[widthOuputLayer];

        for(int idx=0; idx < widthOuputLayer; idx++) {
            Neuron neuron = layers[outputLayerIdx][idx];
            thetas[idx] = (neuron.getOutput() - targets[idx]) * neuron.getOutput() * (1 - neuron.getOutput());
        }

        return thetas;
    }

    private double calculateSigmoidDerivativeTimesOutputPrevLayer(int neuronIdx, int layerIdx, int neuronPrevLayerIdx) {
        return calculateSigmoidDerivative(neuronIdx, layerIdx) * layers[layerIdx - 1][neuronPrevLayerIdx].getOutput();
    }

    private double calculateSigmoidDerivative(int neuronIdx, int layerIdx) {
        Neuron neuron = layers[layerIdx][neuronIdx];
        return neuron.getOutput() * (1 - neuron.getOutput());
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
            layers[0][idx].setInput(inputs[idx]);
        }

        Arrays.stream(layers).forEach(layer -> {
            Arrays.stream(layer).forEach(n->n.fire());
        });
    }

    private double error(double [] targets) {
        double summedError = 0;

        Neuron [] outputLayer = layers[layers.length - 1];

        for(int idx=0; idx < targets.length; idx++) {
            summedError += 0.5 * (targets[idx] - outputLayer[idx].getOutput()) * (targets[idx] - outputLayer[idx].getOutput());
        }

        return summedError;
    }

    public double getOutput(int output) {
        Neuron [] outputLayer = layers[layers.length - 1];
        return outputLayer[output].getOutput();
    }
}
