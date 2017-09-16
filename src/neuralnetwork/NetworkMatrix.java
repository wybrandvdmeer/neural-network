package neuralnetwork;

import Jama.Matrix;

import java.util.*;

public class NetworkMatrix {

    private double learningConstant = 0.5;

    private final String name;

    private Map<Integer, Matrix> weights = new HashMap<>();
    private Map<Integer, Matrix> inputs = new HashMap<>();
    private Map<Integer, Matrix> outputs = new HashMap<>();

    private boolean noTransfer = false;

    public NetworkMatrix(String name, int [] layerSizes) {
        this.name = name;

        for(int layer=0; layer < layerSizes.length - 1; layer++) {
            /* rows = neurons, columns = weights
            */
            Matrix matrix = new Matrix(layerSizes[layer], layerSizes[layer + 1]);
            matrix = matrix.random(matrix.getRowDimension(), matrix.getColumnDimension() + 1); // Plus bias weight.
            matrix = matrix.times(2);
            minus(matrix, 1);
            weights.put(layer, matrix);
        }
    }

    public void setNoTransfer() {
        noTransfer = true;
    }

    public void passForward(double [] input) {
        Matrix inputVector = new Matrix(input, input.length);

        for(int layer = 0; layer < weights.values().size(); layer++) {

            Matrix w = weightsWithoutBias(weights.get(layer));

            inputVector = w.times(inputVector);
            this.inputs.put(layer, inputVector);

            Matrix outputVector = transfer(inputVector);
            this.outputs.put(layer, outputVector);

            inputVector = outputVector;
        }
    }

    private Matrix weightsWithoutBias(Matrix weights) {
        return weights.getMatrix(0, weights.getRowDimension() - 1, 0, weights.getColumnDimension() - 2);
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

    private double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    public double getOutput(int index) {
        return outputs.get(outputs.size() - 1).get(index,0);
    }


















    public int learn(double [] inputs, double [] targets, double errorLimit) throws Exception {
        return learn(inputs, targets, errorLimit, 0);
    }

    public int learn(double [] inputs, double [] targets, double errorLimit, int maxIterations) throws Exception {
        int iterations=0;
        double error=0;

        while(true) {
            //passForward(inputs);

            // error = error(targets);

            if(error < errorLimit) {
                break;
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
                String s = String.format("%.2f", matrix.get(i,j));
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
}
