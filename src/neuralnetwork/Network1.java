package neuralnetwork;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network1 {

    private double learningConstant = 0.5;

    private Neuron [][] layers; // 0 -> input layer.

    private double [][][] weightDerivatives;
    private double [][] biasDerivatives;

    private final String name;

    public Network1(String name, int [] layerSizes) {
        this.name = name;

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

    public Neuron [][] getLayers() {
        return layers;
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

    public int learn(double [] inputs, double [] targets, double errorLimit) throws Exception {
        return learn(inputs, targets, errorLimit, 0);
    }

    public int learn(double [] inputs, double [] targets, double errorLimit, int maxIterations) throws Exception {
        int iterations=0;
        double error;

        while(true) {
            passForward(inputs);

            error = error(targets);

            if(error < errorLimit) {
                break;
            }

            if(maxIterations > 0 && iterations >= maxIterations) {
                String s = String.format("Max iterations exceeded for classifier %s.", name);
                System.out.println(s);
                return -1;
            }

            for (int layerIdx = 1; layerIdx < layers.length; layerIdx++) {
                for (int neuronIdx = 0; neuronIdx < layers[layerIdx].length; neuronIdx++) {

                    Neuron neuron = layers[layerIdx][neuronIdx];

                    List<Double> summationTerms = new ArrayList<>();
                    calculateSummationTerm(layerIdx, neuronIdx, targets, summationTerms, 1);
                    double neuronPd = summationTerms.stream().mapToDouble(d -> d).sum();

                    neuronPd *= this.calculateSigmoidDerivative(layerIdx, neuronIdx);

                    if(layerIdx == layers.length - 1 && neuronIdx == 0) {
                        //System.out.println(String.format("%d-%.16f", iterations, neuronPd));


                        if(iterations >= 17620  && iterations <= 17623) {
                            System.out.println("IT: " + iterations);
                            System.out.println(String.format("W: %.22f - %.22f",
                                    layers[layers.length - 1][0].getWeight(0),
                                    layers[layers.length - 1][0].getWeight(1)));
                            System.out.println(String.format("O1: %.22f", layers[layers.length - 1][0].getOutput()));

                            System.out.println(String.format("H1-out: %.22f", layers[layers.length - 2][0].getOutput()));
                            System.out.println(String.format("H2-out: %.22f", layers[layers.length - 2][1].getOutput()));

                            System.out.println(String.format("T1: %.22f", neuronPd));

                            System.out.println(String.format("GD 2e weight: %.22f",
                                    neuronPd * layers[layerIdx - 1][1].getOutput()));

                            System.out.println(String.format("TD: %.22f",calculateSigmoidDerivative(layerIdx, neuronIdx)));

                        }
                    }





                    for (int weightIdx = 0; weightIdx < neuron.getNoOfWeights(); weightIdx++) {
                        // Times output previous neuralnetwork.Neuron.
                        weightDerivatives[layerIdx - 1][neuronIdx][weightIdx] = neuronPd * layers[layerIdx - 1][weightIdx].getOutput();
                    }

                    biasDerivatives[layerIdx - 1][neuronIdx] = neuronPd;
                }
            }

            for (int layerIdx = 1; layerIdx < layers.length; layerIdx++) {
                for (int neuronIdx = 0; neuronIdx < layers[layerIdx].length; neuronIdx++) {
                    Neuron neuron = layers[layerIdx][neuronIdx];
                    for (int weightIdx = 0; weightIdx < neuron.getNoOfWeights(); weightIdx++) {

                        if(iterations == 17622 && layerIdx == 3) {
                            int i = 0;
                        }

                        adjustWeight(neuron, weightIdx, weightDerivatives[layerIdx - 1][neuronIdx][weightIdx]);
                    }

                    adjustBiasWeight(neuron, biasDerivatives[layerIdx - 1][neuronIdx]);
                }
            }

            iterations++;
        }

        return iterations;
    }

    private void calculateSummationTerm(int layerIdx, int neuronIdx, double [] targets, List<Double> summationTerms, double pd) {
        if(layerIdx < layers.length - 1) {
            for(int neuronInNextLayerIdx=0; neuronInNextLayerIdx < layers[layerIdx + 1].length; neuronInNextLayerIdx++) {
                Neuron neuronInNextLayer = layers[layerIdx + 1][neuronInNextLayerIdx];

                double pdDelta = pd;
                pdDelta *= neuronInNextLayer.getWeight(neuronIdx);
                pdDelta *= calculateSigmoidDerivative(layerIdx + 1, neuronInNextLayerIdx);
                calculateSummationTerm(layerIdx + 1, neuronInNextLayerIdx, targets, summationTerms, pdDelta);
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

    private void write(FileOutputStream weights) throws Exception {
        for(int idx1=1; idx1 < layers.length; idx1++) {
            for(int idx2=0; idx2 < layers[idx1].length; idx2++) {
                for(int idx3=0; idx3 < layers[idx1][idx2].getWeights().length; idx3++) {
                    weights.write((new Double(layers[idx1][idx2].getWeights()[idx3]).toString() + "\n").getBytes());
                }
                weights.write((new Double(layers[idx1][idx2].getBiasWeight()).toString() + "\n").getBytes());
            }
        }
    }

    private void read(FileInputStream weights) throws Exception {
        BufferedReader weightReader = new BufferedReader(new InputStreamReader(weights));

        for(int idx1=1; idx1 < layers.length; idx1++) {
            for(int idx2=0; idx2 < layers[idx1].length; idx2++) {
                for(int idx3=0; idx3 < layers[idx1][idx2].getWeights().length; idx3++) {
                    layers[idx1][idx2].setWeight(idx3, Double.parseDouble(weightReader.readLine()));
                }
                layers[idx1][idx2].setBiasWeight(Double.parseDouble(weightReader.readLine()));
            }
        }
    }

    public String toString() {
        return name;
    }

    public void readWeights() throws Exception {
        File weights = new File(name);
        if(weights.exists()) {
            read(new FileInputStream(weights));
        }
    }

    public void writeWeights() throws Exception {
        File weights = new File(name);
        write(new FileOutputStream(weights));
    }
}

