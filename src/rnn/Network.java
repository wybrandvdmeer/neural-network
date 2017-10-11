package rnn;

import Jama.Matrix;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Network {
    private int timeStamp=0;

    private int noOfOutputs;

    private double learningConstant = 0.1;

    private final String name;

    private Map<Integer, Matrix> weights = new HashMap<>();
    private Map<Integer, Matrix> biasWeights = new HashMap<>();

    private List<Map<Integer, Matrix>> outputsPerTimestamp = new ArrayList<>();
    private List<Map<Integer, Matrix>> transferDerivertivesPerTimestamp = new ArrayList<>(); // E.g. dNout/dNin

    private Map<Integer, Map<Integer, Matrix>> gradientsPerLayerPerOutput = new HashMap<>();
    private Map<Integer, Map<Integer, Matrix>> biasGradientsPerLayerPerOutput = new HashMap<>();

    // Weights between the first hidden layer of different timestamps.
    private Matrix W;

    private double error;

    private boolean noTransfer=false;

    private final double RELU_LEAKAGE = 0.1;

    public Network(String name, int [] layerSizes, int noOfTimeSteps) {
        this.name = name;
        this.noOfOutputs = noOfTimeSteps;

        for(int layer=0; layer < layerSizes.length - 1; layer++) {
            /* rows = neurons, columns = weights
            */
            Matrix weightsPerLayer = new Matrix(layerSizes[layer+1], layerSizes[layer]);
            weightsPerLayer = initializeWeights(weightsPerLayer);

            Matrix biasWeightsPerLayer = new Matrix(layerSizes[layer + 1], 1);
            biasWeightsPerLayer = initializeWeights(biasWeightsPerLayer);

            weights.put(layer, weightsPerLayer);
            biasWeights.put(layer, biasWeightsPerLayer);
        }

        for(int output=0; output < noOfTimeSteps; output++) {
            gradientsPerLayerPerOutput.put(output, new HashMap<>());
            biasGradientsPerLayerPerOutput.put(output, new HashMap<>());
            for (int layer = 0; layer < layerSizes.length - 1; layer++) {
                gradientsPerLayerPerOutput.get(output).put(layer, weights.get(layer).copy().times(0));
                biasGradientsPerLayerPerOutput.get(output).put(layer, biasWeights.get(layer).copy().times(0));
            }
        }

        W = new Matrix(layerSizes[1], layerSizes[1]);
        W = initializeWeights(W);
    }

    public void setNoTransfer() {
        noTransfer = true;
    }

    public void passForward(double [][] inputs) {
        if(inputs.length != noOfOutputs) {
            throw new RuntimeException("Wrong dimensions.");
        }

        timeStamp = 0;

        for(int i=0; i < inputs.length; i++) {
            passForward(inputs[i]);
            nextTimestamp();
        }
    }

    public void passForward(double [] input) {
        Matrix inputVector = new Matrix(input, input.length);
        storePerTimestamp(0, inputVector, outputsPerTimestamp);

        for(int layer = 1; layer <= weights.values().size(); layer++) {
            inputVector = weights.get(layer - 1).times(inputVector);
            inputVector = inputVector.plus(biasWeights.get(layer - 1));

            if(layer == 1 && timeStamp > 0) {
                inputVector = inputVector.plus(W.times(outputsPerTimestamp.get(timeStamp - 1).get(1)));
            }

            boolean hidden = layer < weights.values().size();

            Matrix outputVector = transfer(inputVector, hidden);

            storePerTimestamp(layer, outputVector, outputsPerTimestamp);
            storePerTimestamp(layer, get2Dim(transferDerivative(outputVector, hidden)), transferDerivertivesPerTimestamp);

            inputVector = outputVector;
        }
    }

    private Matrix transferDerivative(Matrix vector, boolean hidden) {
        Matrix v2 = vector.copy();

        if(noTransfer) {
            for(int row=0; row < vector.getRowDimension(); row++) {
                v2.set(row, 0, 1);
            }
            return v2;
        }

        if(hidden) {
            for(int row=0; row < vector.getRowDimension(); row++) {
                if(vector.get(row, 0) >= 0) {
                    v2.set(row, 0, 1);
                } else {
                    v2.set(row, 0, RELU_LEAKAGE);
                }
            }
            return v2;
        }

        for(int row=0; row < vector.getRowDimension(); row++) {
            v2.set(row, 0, vector.get(row, 0) * (1 - vector.get(row, 0)));
        }

        return v2;
    }

    private Matrix transfer(Matrix vector, boolean hidden) {
        Matrix transfered = new Matrix(vector.getRowDimension(), 1);

        if(noTransfer) {
            for (int kol = 0; kol < vector.getColumnDimension(); kol++) {
                for (int row = 0; row < vector.getRowDimension(); row++) {
                    transfered.set(row, kol, vector.get(row, kol));
                }
            }
            return transfered;
        }

        if(hidden) {
            for (int kol = 0; kol < vector.getColumnDimension(); kol++) {
                for (int row = 0; row < vector.getRowDimension(); row++) {
                    if(vector.get(row, kol) >= 0) {
                        transfered.set(row, kol, vector.get(row, kol));
                    } else {
                        transfered.set(row, kol, RELU_LEAKAGE * vector.get(row, kol));
                    }
                }
            }
            return transfered;
        }

        for (int kol = 0; kol < vector.getColumnDimension(); kol++) {
            for (int row = 0; row < vector.getRowDimension(); row++) {
                transfered.set(row, kol, sigmoid(vector.get(row, kol)));
            }
        }

        return transfered;
    }

    public int learn(double [][] inputs, double [][] targets, double errorLimit, int maxIterations) throws Exception {

        if(inputs.length != targets.length && inputs.length != noOfOutputs) {
            throw new RuntimeException("Wrong dimensions.");
        }

        int iterations=0;

        while(true) {
            error = 0;

            for(int output=0; output < targets.length; output++) {
                passForward(inputs[output]);
                Matrix targetVector = new Matrix(targets[output], targets[output].length);
                error += error(output, targetVector);
                nextTimestamp();
            }

            System.out.println("Error: " + error);

            if(error < errorLimit) {
                break;
            }

            for(int output = 0; output < noOfOutputs; output++) {
                for (int layer : gradientsPerLayerPerOutput.get(output).keySet()) {
                    gradientsPerLayerPerOutput.get(output).put(layer, gradientsPerLayerPerOutput.get(output).get(layer).times(0));
                }

                for (int layer : biasGradientsPerLayerPerOutput.get(output).keySet()) {
                    biasGradientsPerLayerPerOutput.get(output).put(layer, biasGradientsPerLayerPerOutput.get(output).get(layer).times(0));
                }
            }

            Matrix wGradients = W.copy().times(0);

            Matrix thetaTime = null;

            for(int output=targets.length - 1; output >= 0; output--) {

                Matrix outputVector = outputsPerTimestamp.get(output).get(outputsPerTimestamp.get(output).size() - 1);
                Matrix errorDeriv = outputVector.minus(new Matrix(targets[output], targets[output].length));
                Matrix theta = null;

                for (int layer = weights.values().size(); layer > 1; layer--) {
                    Matrix transferDerivatives = transferDerivertivesPerTimestamp.get(output).get(layer);
                    if(theta == null) {
                        theta = transferDerivatives.times(errorDeriv).transpose();
                    } else {
                        theta = theta.times(weights.get(layer).times(transferDerivatives));
                    }

                    Map<Integer, Matrix> outputs = outputsPerTimestamp.get(output);

                    gradientsPerLayerPerOutput.get(output).put(layer - 1, gradientsPerLayerPerOutput.get(output).get(layer - 1).plus(outputs.get(layer - 1).times(theta).transpose()));
                    biasGradientsPerLayerPerOutput.get(output).put(layer - 1, biasGradientsPerLayerPerOutput.get(output).get(layer - 1).plus(theta.transpose()));
                }

                /*

                for(int timeStep=noOfOutputs - 1; timeStep >= 0; timeStep--) {

                    Matrix errorDeriv = getOutputVector().minus(new Matrix(targets[timeStep], targets[timeStep].length));
                    Matrix theta = null;

                    for (int layer = weights.values().size(); layer > 0; layer--) {
                        Matrix transferDerivatives = transferDerivertivesPerTimestamp.get(timeStep).get(layer);
                        if(theta == null) {
                            theta = transferDerivatives.times(errorDeriv).transpose();
                        } else {
                            theta = theta.times(weights.get(layer).times(transferDerivatives));
                        }

                        Map<Integer, Matrix> outputs = outputsPerTimestamp.get(timeStep);

                        if(layer > 1) {
                            gradientsPerLayer.put(layer, gradientsPerLayer.get(layer).plus(outputs.get(layer - 1).times(theta).transpose()));
                            biasGradientsPerLayer.put(layer, biasGradientsPerLayer.get(layer).plus(theta.transpose()));
                        } else {
                            if(thetaTime == null) {
                                thetaTime = theta;
                            } else {
                                thetaTime = thetaTime.times(W);
                            }

                            gradientsPerLayer.put(layer, gradientsPerLayer.get(layer).plus(outputs.get(layer - 1).times(thetaTime).transpose()));
                            biasGradientsPerLayer.put(layer, biasGradientsPerLayer.get(layer).plus(thetaTime.transpose()));

                            if(timeStep > 0) {
                                wGradients = wGradients.plus(outputsPerTimestamp.get(timeStep - 1).get(0).times(thetaTime).transpose());
                            }
                        }
                    }
                    */
            }

            for(int output = 0; output < noOfOutputs; output++) {
                for (int layer = weights.values().size(); layer > 0; layer--) {
                    weights.put(layer - 1, weights.get(layer - 1).minus(gradientsPerLayerPerOutput.get(output).get(layer - 1).times(learningConstant)));
                    biasWeights.put(layer - 1, biasWeights.get(layer - 1).minus(biasGradientsPerLayerPerOutput.get(output).get(layer - 1).times(learningConstant)));
                }
            }

            W = W.minus(wGradients.times(learningConstant));

            iterations++;

            if(maxIterations > 0 && iterations >= maxIterations) {
                String s = String.format("Max iterations exceeded for classifier %s.", name);
                System.out.println(s);
                return -1;
            }
        }

        return iterations;
    }

    Map<Integer, Matrix> getGradients(int output) {
        return gradientsPerLayerPerOutput.get(output);
    }

    Map<Integer, Matrix> getBiasGradients(int output) {
        return biasGradientsPerLayerPerOutput.get(output);
    }

    double error(int output, Matrix targets) {
        double error=0;
        Matrix m1 = targets.minus(getOutputVector(output));
        for(int row=0; row < m1.getRowDimension(); row++) {
            error += m1.get(row, 0) * m1.get(row, 0) * 0.5;
        }

        return error;
    }

    private void minus(Matrix m, double minus) {
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                m.set(i, j, m.get(i, j) - minus);
            }
        }
    }
    public void printMatrix(Matrix matrix) {
        printMatrix(matrix, "matrix");
    }

    public void printMatrix(Matrix matrix, String name) {
        System.out.println("Matrix: " + name);
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                String s = String.format("%.8f", matrix.get(i,j));
                System.out.print(' ');
                System.out.print(s);
            }
            System.out.println();
        }
        System.out.println();
    }

    public Matrix getWeights(int layer) {
        return weights.get(layer - 1);
    }

    public Matrix getW() {
        return W;
    }

    public Matrix getBiasWeights(int layer) {
        return biasWeights.get(layer - 1);
    }

    private double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    private Matrix getOutputVector(int output) {
        return outputsPerTimestamp.get(output).get(outputsPerTimestamp.get(output).size() - 1);
    }

    public double getOutput(int index) {
        return getOutputVector(noOfOutputs - 1).get(index,0);
    }

    private Matrix get2Dim(Matrix vector) {
        Matrix matrix = new Matrix(vector.getRowDimension(), vector.getRowDimension());

        for(int row=0; row < vector.getRowDimension(); row++) {
            for(int col=0; col < vector.getRowDimension(); col++) {
                if(col == row) {
                    matrix.set(row, col, vector.get(row, 0));
                } else {
                    matrix.set(row, col, 0);
                }
            }
        }

        return matrix;
    }

    public double getError() {
        return error;
    }

    private double getL2Norm(Matrix matrix) {
        double l2=0;
        for(int row=0; row < matrix.getRowDimension(); row++) {
            for(int col=0; col < matrix.getColumnDimension(); col++) {
                l2 += (matrix.get(row, col) * matrix.get(row, col));
            }
        }
        return Math.sqrt(l2);
    }


    /* The floating point arithmetic introduces a small errors. In order to avoid this error,
    we only store a mantissa of 8 digits.
    */
    public void write() throws Exception {
        FileOutputStream weightsFile = new FileOutputStream(name);
        for(int layer : weights.keySet()) {
            for(int row=0; row < weights.get(layer).getRowDimension(); row++) {
                for(int col=0; col < weights.get(layer).getColumnDimension(); col++) {
                    weightsFile.write((String.format("%.8f", weights.get(layer).get(row,col)) + "\n").getBytes());
                }
                weightsFile.write((String.format("%.8f", biasWeights.get(layer).get(row, 0)) + "\n").getBytes());
            }
        }
    }

    public void read() throws Exception {
        File file = new File(name);
        if(!file.exists()) {
            return;
        }

        BufferedReader weightReader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));

        for(int layer : weights.keySet()) {
            for(int row=0; row < weights.get(layer).getRowDimension(); row++) {
                for(int col=0; col < weights.get(layer).getColumnDimension(); col++) {
                    weights.get(layer).set(row, col, Double.parseDouble(weightReader.readLine()));
                }
                biasWeights.get(layer).set(row, 0, Double.parseDouble(weightReader.readLine()));
            }
        }
    }

    private void storePerTimestamp(int layer, Matrix vector, List<Map<Integer, Matrix>> list) {
        if(list.size() <= timeStamp) {
            Map<Integer, Matrix> matrixPerLayer = new HashMap<>();
            list.add(timeStamp, matrixPerLayer);
        }

        list.get(timeStamp).put(layer, vector);
    }

    private void dropOffOneElement(List<Map<Integer, Matrix>> list) {
        System.arraycopy(list.toArray(), 1, list.toArray(), 0, list.toArray().length - 1);
    }

    public void nextTimestamp() {
        if(++timeStamp >= noOfOutputs) {
            dropOffOneElement(outputsPerTimestamp);
            dropOffOneElement(transferDerivertivesPerTimestamp);
            timeStamp = noOfOutputs - 1;
        }
    }

    private Matrix initializeWeights(Matrix matrix) {
        Matrix initializedWeights = matrix.copy();
        initializedWeights = initializedWeights.random(initializedWeights.getRowDimension(), initializedWeights.getColumnDimension());
        initializedWeights = initializedWeights.times(2);
        minus(initializedWeights, 1);
        return initializedWeights;
    }
}
