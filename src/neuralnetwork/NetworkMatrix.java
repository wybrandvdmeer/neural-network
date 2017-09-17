package neuralnetwork;

import Jama.Matrix;

import java.util.*;

public class NetworkMatrix {

    private double learningConstant = 0.5;

    private final String name;

    private Map<Integer, Matrix> weights = new HashMap<>();
    private Map<Integer, Matrix> outputs = new HashMap<>();
    private Map<Integer, Matrix> transferDerivertives = new HashMap<>(); // E.g. dNout/dNin
    private Map<Integer, Matrix> gradients = new HashMap<>();

    private double error;

    private boolean noTransfer = false;

    public NetworkMatrix(String name, int [] layerSizes) {
        this.name = name;

        for(int layer=0; layer < layerSizes.length - 1; layer++) {
            /* rows = neurons, columns = weights
            */
            Matrix weightsPerLayer = new Matrix(layerSizes[layer], layerSizes[layer + 1]);
            weightsPerLayer = weightsPerLayer.random(weightsPerLayer.getRowDimension(), weightsPerLayer.getColumnDimension() + 1); // Plus bias weight.
            weightsPerLayer = weightsPerLayer.times(2);
            minus(weightsPerLayer, 1);
            weights.put(layer, weightsPerLayer);
        }
    }

    public void setNoTransfer() {
        noTransfer = true;
    }

    public void passForward(double [] input) {
        Matrix inputVector = new Matrix(input.length + 1, 1);

        int row=0;
        for(; row < input.length; row++) {
            inputVector.set(row, 0, input[row]);
        }

        inputVector.set(row, 0, 1); // bias.

        // Store the input as output of the input layer.
        outputs.put(0, inputVector.getMatrix(0, input.length - 1, 0, 0));

        for(int layer = 1; layer <= weights.values().size(); layer++) {

            inputVector = weights.get(layer - 1).times(inputVector);

            Matrix outputVector = transfer(inputVector);
            outputs.put(layer, outputVector);

            transferDerivertives.put(layer, get2Dim(transferDerivative(outputVector)));

            inputVector = new Matrix(outputVector.getRowDimension() + 1, 1);
            for(row=0; row < inputVector.getRowDimension() -1; row++) {
                inputVector.set(row, 0, outputVector.get(row, 0));
            }

            inputVector.set(row, 0, 1); // Bias.
        }
    }

    private Matrix transferDerivative(Matrix vector) {
        Matrix v2 = vector.copy();
        for(int row=0; row < vector.getRowDimension(); row++) {
            if(noTransfer) {
                v2.set(row, 0, 1);
            } else {
                v2.set(row, 0, vector.get(row, 0) * (1 - vector.get(row, 0)));
            }
        }

        return v2;
    }

    private Matrix transfer(Matrix vector) {
        Matrix transfered = new Matrix(vector.getRowDimension(), 1);

        for(int kol=0; kol < vector.getColumnDimension(); kol++) {
            for(int row=0; row < vector.getRowDimension(); row++) {
                if(noTransfer) {
                    transfered.set(row, kol, vector.get(row, kol));
                } else {
                    transfered.set(row, kol, sigmoid(vector.get(row, kol)));
                }
            }
        }
        return transfered;
    }

    public int learn(double [] inputs, double [] targets, double errorLimit, int maxIterations) throws Exception {
        int iterations=0;

        Matrix targetVector = new Matrix(targets, targets.length);

        while(true) {
            passForward(inputs);

            error = error(targetVector);

            if(error < errorLimit) {
                break;
            }

            Matrix errorDeriv = getOutputVector().minus(targetVector);

            Matrix theta;

            for(int layer=weights.values().size(); layer > 0; layer--) {
                theta = outputs.get(layer - 1).times(transferDerivertives.get(layer).times(errorDeriv).transpose()).transpose();
                gradients.put(layer, theta);
            }

            if(maxIterations > 0 && iterations >= maxIterations) {
                String s = String.format("Max iterations exceeded for classifier %s.", name);
                System.out.println(s);
                return -1;
            }

            iterations++;
        }

        return iterations;
    }

    public int learn(double [] inputs, double [] targets, double errorLimit) throws Exception {
        return learn(inputs, targets, errorLimit, 0);
    }

    Matrix getGradients(int layer) {
        return gradients.get(layer);
    }

    double error(Matrix targets) {
        double error=0;
        Matrix m1 = targets.minus(getOutputVector());
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

    public void printMatrix(Matrix matrix, String name) {
        System.out.println("Matrix: " + name);
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                String s = String.format("%.4f", matrix.get(i,j));
                System.out.print(' ');
                System.out.print(s);
            }
            System.out.println();
        }
        System.out.println();
    }

    public Map<Integer, Matrix> getWeights() {
        return weights;
    }

    private double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    private Matrix getOutputVector() {
        return outputs.get(outputs.size() - 1);
    }

    public double getOutput(int index) {
        return getOutputVector().get(index,0);
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
}
