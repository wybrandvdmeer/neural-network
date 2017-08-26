import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ScalableLengthNetwork {

    private double learningConstant = 0.5;

    private Neuron [][] layers; // 0 -> input layer.

    private double [][][] weightDerivatives;
    private double [][] biasDerivatives;

    public ScalableLengthNetwork(int [] layerSizes) {
        layers = new Neuron[layerSizes.length][];
        weightDerivatives = new double[layerSizes.length - 1][][];
        biasDerivatives = new double[layerSizes.length - 1][];

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
            biasDerivatives[idx1 - 1] = new double[layerSizes[idx1]];
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
        learn(inputs, targets, errorLimit, 0);
    }

    public void learn(double [] inputs, double [] targets, double errorLimit, int maxIterations) {
        int iterations=0;
        double error;

        while(true) {
            passForward(inputs);

            error = error(targets);

            System.out.println(String.format("It: %d. Error: %f", iterations, error));

            if(error < errorLimit || (maxIterations > 0 && iterations >= maxIterations)) {
                break;
            }

            for (int layerIdx = 1; layerIdx < layers.length; layerIdx++) {
                for (int neuronIdx = 0; neuronIdx < layers[layerIdx].length; neuronIdx++) {
                    Neuron neuron = layers[layerIdx][neuronIdx];

                    List<Double> summationTerms = new ArrayList<>();
                    calculateSummationTerm(layerIdx, neuronIdx, targets, summationTerms, 1);
                    double neuronPd = summationTerms.stream().mapToDouble(d -> d).sum();

                    neuronPd *= this.calculateSigmoidDerivative(layerIdx, neuronIdx);

                    for (int weightIdx = 0; weightIdx < neuron.getNoOfWeights(); weightIdx++) {
                        // Times output previous Neuron.
                        weightDerivatives[layerIdx - 1][neuronIdx][weightIdx] = neuronPd * layers[layerIdx - 1][weightIdx].getOutput();
                    }

                    biasDerivatives[layerIdx - 1][neuronIdx] = neuronPd;
                }
            }

            for (int layerIdx = 1; layerIdx < layers.length; layerIdx++) {
                for (int neuronIdx = 0; neuronIdx < layers[layerIdx].length; neuronIdx++) {
                    Neuron neuron = layers[layerIdx][neuronIdx];
                    for (int weightIdx = 0; weightIdx < neuron.getNoOfWeights(); weightIdx++) {
                        adjustWeight(neuron, weightIdx, weightDerivatives[layerIdx - 1][neuronIdx][weightIdx]);
                    }

                    adjustBiasWeight(neuron, biasDerivatives[layerIdx - 1][neuronIdx]);
                }
            }

            iterations++;
        }
    }

    private void calculateSummationTerm(int layerIdx, int neuronIdx, double [] targets, List<Double> summationTerms, double pd) {
        if(layerIdx < layers.length - 1) {
            for(int neuronInNextLayerIdx=0; neuronInNextLayerIdx < layers[layerIdx + 1].length; neuronInNextLayerIdx++) {
                Neuron neuronInNextLayer = layers[layerIdx + 1][neuronInNextLayerIdx];
                pd *= neuronInNextLayer.getWeight(neuronIdx);
                pd *= calculateSigmoidDerivative(layerIdx + 1, neuronInNextLayerIdx);
                calculateSummationTerm(layerIdx + 1, neuronInNextLayerIdx, targets, summationTerms, pd);
            }
        } else {
            // When arriving at the output, calculate dE/dOoutput
            Neuron outputNeuron = layers[layers.length - 1][neuronIdx];
            pd *= (outputNeuron.getOutput() - targets[neuronIdx]);
            summationTerms.add(pd);
        }
    }

    private double calculateSigmoidDerivative(int layerIdx, int neuronIdx) {
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
