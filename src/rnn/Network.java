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

    private double learningRate;

    private final double GRADIENT_CLIPPING_TRESHOLD = 1;

    private final String name;

    private Map<Integer, Matrix> weights = new HashMap<>();
    private Map<Integer, Matrix> biasWeights = new HashMap<>();

    // Weights between the first hidden layer of different timestamps.
    private Matrix W;

    private List<Map<Integer, Matrix>> outputsPerTimestamp = new ArrayList<>();
    private List<Map<Integer, Matrix>> transferDerivertivesPerTimestamp = new ArrayList<>(); // E.g. dNout/dNin

    private Map<Integer, Map<Integer, Matrix>> gradientsPerLayerPerOutput = new HashMap<>();
    private Map<Integer, Map<Integer, Matrix>> biasGradientsPerLayerPerOutput = new HashMap<>();
    private Map<Integer, Matrix> wGradientsPerOutput = new HashMap<>();

    private double error;

    // Var indicates from which output (timestamp) to count the error.
    private int beginErrorOutput=0;

    private String weightFileDir=null;

    private boolean tanh=false;

    public Network(String name, int [] layerSizes, int noOfOutputs) {
        this(name, layerSizes, noOfOutputs, false);
    }

    public Network(String name, int [] layerSizes, int noOfOutputs, boolean tanh) {
        this.name = name;
        this.noOfOutputs = noOfOutputs;

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

        W = new Matrix(layerSizes[1], layerSizes[1]);
        W = initializeWeights(W);

        for(int output=0; output < noOfOutputs; output++) {
            gradientsPerLayerPerOutput.put(output, new HashMap<>());
            biasGradientsPerLayerPerOutput.put(output, new HashMap<>());

            for (int layer = 0; layer < layerSizes.length - 1; layer++) {
                gradientsPerLayerPerOutput.get(output).put(layer,
                        new Matrix(weights.get(layer).getRowDimension(),
                        weights.get(layer).getColumnDimension()));
                biasGradientsPerLayerPerOutput.get(output).put(layer,
                        new Matrix(biasWeights.get(layer).getRowDimension(), 1));
            }

            wGradientsPerOutput.put(output, new Matrix(W.getRowDimension(), W.getColumnDimension()));
        }
    }

    public void setWeightFileDir(String weightFileDir) {
        this.weightFileDir = weightFileDir;
    }

    public void setBeginErrorOutput(int beginErrorOutput) {
        this.beginErrorOutput = beginErrorOutput;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void passForward(double [][] inputs) {
        if(inputs.length != noOfOutputs) {
            throw new RuntimeException("Wrong dimensions.");
        }

        timeStamp = 0;

        for(int i=0; i < inputs.length; i++) {
            if(i == 0) {
                passForward(inputs[i]);
            } else {
                passForward(inputs[i]);
            }
            nextTimestamp();
        }
    }

    public void passForward(double [] input) {
        Matrix inputVector = new Matrix(input, input.length);
        storePerTimestamp(0, inputVector, outputsPerTimestamp);

        for(int layer = 1; layer <= weights.values().size(); layer++) {
            inputVector = weights.get(layer - 1).times(inputVector);
            inputVector = inputVector.plus(biasWeights.get(layer - 1));

            if(layer == 1 && timeStamp > 0){
                inputVector = inputVector.plus(W.times(outputsPerTimestamp.get(timeStamp - 1).get(1)));
            }

            Matrix outputVector = transfer(inputVector, layer == weights.values().size());

            storePerTimestamp(layer, outputVector, outputsPerTimestamp);
            storePerTimestamp(layer, get2Dim(transferDerivative(outputVector, layer == weights.values().size())), transferDerivertivesPerTimestamp);

            inputVector = outputVector;
        }
    }

    private Matrix transferDerivative(Matrix vector, boolean outputLayer) {
        Matrix v2 = vector.copy();

        /* This derivative is ignored, because in the case of softmax dE/dOin is calculated
        instead of dE/dOout * dOout/dOin.
        */
        if(outputLayer) {
            return v2;
        }

        if(tanh) {
            for(int row=0; row < vector.getRowDimension(); row++) {
                v2.set(row, 0, tanhDeriv(vector.get(row, 0)));
            }
            return v2;
        }

        for(int row=0; row < vector.getRowDimension(); row++) {
            v2.set(row, 0, vector.get(row, 0) * (1 - vector.get(row, 0)));
        }

        return v2;
    }

    private Matrix transfer(Matrix vector, boolean outputLayer) {
        Matrix transfered = new Matrix(vector.getRowDimension(), 1);

        if(outputLayer) {
            double sum=0;
            for (int row = 0; row < vector.getRowDimension(); row++) {
                transfered.set(row, 0, Math.pow(Math.E, vector.get(row, 0)));
                sum += transfered.get(row, 0);
            }
            for (int row = 0; row < transfered.getRowDimension(); row++) {
                transfered.set(row, 0, transfered.get(row, 0) / sum);
            }
            return transfered;
        }

        if(tanh) {
            for (int kol = 0; kol < vector.getColumnDimension(); kol++) {
                for (int row = 0; row < vector.getRowDimension(); row++) {
                    transfered.set(row, kol, tanh(vector.get(row, kol)));
                }
            }
        } else {
            for (int kol = 0; kol < vector.getColumnDimension(); kol++) {
                for (int row = 0; row < vector.getRowDimension(); row++) {
                    transfered.set(row, kol, sigmoid(vector.get(row, kol)));
                }
            }
        }

        return transfered;
    }

    public int learn(double [][] inputs, double [][] targets, double errorLimit, int maxIterations) throws Exception {

        if(inputs.length != targets.length || inputs.length != noOfOutputs) {
            throw new RuntimeException("Wrong dimensions.");
        }

        int iterationsWhenErrorIsIncreasing=0;
        int learningConstantReductions=0;
        double previousError=100;

        learningRate = 0.1;

        int iterations=0;

        while(true) {
            error = 0;

            passForward(inputs);

            for(int output=beginErrorOutput; output < targets.length; output++) {
                error += error(output, new Matrix(targets[output], targets[output].length));
            }

            if(error < errorLimit) {
                break;
            }

            if(previousError < error) {
                iterationsWhenErrorIsIncreasing = iterations;

                if(learningRate > 0.005) {
                    System.out.println(String.format("Error is getting bigger (%f -> %f): adjusted learningRate (%f -> %f)",
                            previousError, error, learningRate, learningRate * 0.5));

                    learningConstantReductions++;
                    learningRate *= 0.5;
                }
            }

            if(learningConstantReductions > 0 && iterations - iterationsWhenErrorIsIncreasing > 100000) {
                System.out.println("Increasing learning rate: " + learningRate * 2);
                learningRate *= 2;
                learningConstantReductions--;
                iterationsWhenErrorIsIncreasing = iterations - 50000;
            }

            if(iterations%10000 == 0) {
                System.out.println("Error: " + error);
            }

            previousError = error;

            for(int output = 0; output < noOfOutputs; output++) {
                for (int layer : gradientsPerLayerPerOutput.get(output).keySet()) {
                    gradientsPerLayerPerOutput.get(output).put(layer, initMatrix(gradientsPerLayerPerOutput.get(output).get(layer)));
                }

                for (int layer : biasGradientsPerLayerPerOutput.get(output).keySet()) {
                    biasGradientsPerLayerPerOutput.get(output).put(layer, initMatrix(biasGradientsPerLayerPerOutput.get(output).get(layer)));
                }

                wGradientsPerOutput.put(output, initMatrix(W.copy()));
            }

            for(int output=targets.length - 1; output >= 0; output--) {

                Matrix outputVector = outputsPerTimestamp.get(output).get(outputsPerTimestamp.get(output).size() - 1);
                Matrix errorDeriv = outputVector.minus(new Matrix(targets[output], targets[output].length));
                Matrix theta = null;
                Matrix thetaTime = null;

                for (int layer = weights.values().size(); layer > 0; layer--) {
                    Matrix transferDerivatives = transferDerivertivesPerTimestamp.get(output).get(layer);
                    if(theta == null) {
                        theta = errorDeriv.transpose();
                    } else {
                        theta = theta.times(weights.get(layer).times(transferDerivatives));
                    }

                    Map<Integer, Matrix> outputs = outputsPerTimestamp.get(output);

                    if(layer > 1) {
                        gradientsPerLayerPerOutput.get(output).put(layer - 1, gradientsPerLayerPerOutput.get(output).get(layer - 1).plus(outputs.get(layer - 1).times(theta).transpose()));
                        biasGradientsPerLayerPerOutput.get(output).put(layer - 1, biasGradientsPerLayerPerOutput.get(output).get(layer - 1).plus(theta.transpose()));
                    }
                }

                for(int timeStep=output; timeStep >= 0; timeStep--) {

                    if (thetaTime == null) {
                        thetaTime = theta;
                    } else {
                        Matrix transferDerivatives = transferDerivertivesPerTimestamp.get(output - 1).get(1);
                        thetaTime = thetaTime.times(W.times(transferDerivatives));
                    }

                    Map<Integer, Matrix> outputs = outputsPerTimestamp.get(output);

                    gradientsPerLayerPerOutput.get(output).put(0,
                            gradientsPerLayerPerOutput.get(output).get(0).plus(outputs.get(0).times(thetaTime).transpose()));
                    biasGradientsPerLayerPerOutput.get(output).put(0,
                            biasGradientsPerLayerPerOutput.get(output).get(0).plus(thetaTime.transpose()));

                    if (timeStep > 0) {
                        wGradientsPerOutput.put(output, wGradientsPerOutput.get(output).plus(outputsPerTimestamp.get(timeStep - 1).get(1).times(thetaTime).transpose()));
                    }
                }
            }

            for (int layer = weights.values().size(); layer > 0; layer--) {
                Matrix gradients = initMatrix(weights.get(layer - 1).copy());
                Matrix biasGradients = initMatrix(biasWeights.get(layer - 1).copy());

                for(int output = 0; output < noOfOutputs; output++) {
                    gradients = gradients.plus(gradientsPerLayerPerOutput.get(output).get(layer - 1));
                    biasGradients = biasGradients.plus(biasGradientsPerLayerPerOutput.get(output).get(layer - 1));
                }

                weights.put(layer - 1, weights.get(layer - 1).minus(gradients.times(learningRate)));
                biasWeights.put(layer - 1, biasWeights.get(layer - 1).minus(biasGradients.times(learningRate)));
            }

            Matrix wGradients = initMatrix(W.copy());
            for(int output = 0; output < noOfOutputs; output++) {
                wGradients = wGradients.plus(wGradientsPerOutput.get(output));
            }

            W = W.minus(wGradients.times(learningRate));

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

    Matrix getWGradients(int output) {
        return wGradientsPerOutput.get(output);
    }

    Map<Integer, Matrix> getBiasGradients(int output) {
        return biasGradientsPerLayerPerOutput.get(output);
    }

    List<Map<Integer, Matrix>> getTransferDerivertivesPerTimestamp() {
        return transferDerivertivesPerTimestamp;
    }

    double error(int output, Matrix targets) {
        double error=0;
        for(int row=0; row < getOutputVector(output).getRowDimension(); row++) {
            error += targets.get(row, 0) * Math.log(getOutputVector(output).get(row, 0));
        }
        return -1 * error;
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

    private double tanh(double x) {
        return (Math.pow(Math.E, x) - Math.pow(Math.E, -1 * x) ) / (Math.pow(Math.E, x) + Math.pow(Math.E, -1 * x));
    }

    private double tanhDeriv(double x) {
        double numerator = Math.pow(Math.E, x) - Math.pow(Math.E, -1 * x);
        double denominator = Math.pow(Math.E, x) + Math.pow(Math.E, -1 * x);
        return 1 - (numerator * numerator) / (denominator * denominator);
    }

    public Matrix getOutputVector(int output) {
        return outputsPerTimestamp.get(output).get(outputsPerTimestamp.get(output).size() - 1);
    }

    public double getOutput(int index) {
        return getOutputVector(noOfOutputs - 1).get(index, 0);
    }

    public Matrix getOutputVector() {
        return outputsPerTimestamp.get(noOfOutputs - 1).get(outputsPerTimestamp.get(noOfOutputs - 1).size() - 1);
    }

    public Matrix getHiddenState(int output) {
        return outputsPerTimestamp.get(output).get(1);
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


    /* The floating point arithmetic introduces a small errors. In order to avoid this error,
    we only store a mantissa of 8 digits.
    */
    public void write() throws Exception {
        File weightsFile = new File(weightFileDir, name);
        FileOutputStream out = new FileOutputStream(weightsFile);
        for(int layer : weights.keySet()) {
            for(int row=0; row < weights.get(layer).getRowDimension(); row++) {
                for(int col=0; col < weights.get(layer).getColumnDimension(); col++) {
                    out.write((String.format("%.8f", weights.get(layer).get(row,col)) + "\n").getBytes());
                }
                out.write((String.format("%.8f", biasWeights.get(layer).get(row, 0)) + "\n").getBytes());
            }
        }

        for(int row=0; row < W.getRowDimension(); row++) {
            for (int col = 0; col < W.getColumnDimension(); col++) {
                out.write((String.format("%.8f", W.get(row, col)) + "\n").getBytes());
            }
        }
    }

    public void read() throws Exception {
        File file = new File(weightFileDir, name);
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

        for(int row=0; row < W.getRowDimension(); row++) {
            for (int col = 0; col < W.getColumnDimension(); col++) {
                W.set(row, col, Double.parseDouble(weightReader.readLine()));
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

    private Matrix initMatrix(Matrix matrix) {
        for(int row=0; row < matrix.getRowDimension(); row++) {
            for(int col=0; col < matrix.getColumnDimension(); col++) {
                matrix.set(row, col, 0);
            }
        }
        return matrix;
    }
}
