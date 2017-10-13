package nn;

import Jama.Matrix;

import java.io.*;
import java.util.*;

public class Network {

    private final double GRADIENT_CLIPPING_TRESHOLD = 1;

    private double learningConstant = 0.5;

    private final String name;

    private Map<Integer, Matrix> weights = new HashMap<>();
    private Map<Integer, Matrix> biasWeights = new HashMap<>();
    private Map<Integer, Matrix> outputs = new HashMap<>();
    private Map<Integer, Matrix> transferDerivertives = new HashMap<>(); // E.g. dNout/dNin
    private Map<Integer, Matrix> gradientsPerLayer = new HashMap<>();
    private Map<Integer, Matrix> biasGradientsPerLayer = new HashMap<>();

    private double error;

    private boolean noTransfer=false, leakyRelu=false, gradientClipping=true;

    private final double RELU_LEAKAGE = 0.1;

    public Network(String name, int [] layerSizes) {
        this(name, layerSizes, false);
    }

    public Network(String name, int [] layerSizes, boolean leakyRelu) {
        this.name = name;
        this.leakyRelu = leakyRelu;

        for(int layer=0; layer < layerSizes.length - 1; layer++) {
            /* rows = neurons, columns = weights
            */
            Matrix weightsPerLayer = new Matrix(layerSizes[layer+1], layerSizes[layer]);
            weightsPerLayer = weightsPerLayer.random(weightsPerLayer.getRowDimension(), weightsPerLayer.getColumnDimension());
            weightsPerLayer = weightsPerLayer.times(2);
            minus(weightsPerLayer, 1);

            Matrix biasWeightsPerLayer = new Matrix(layerSizes[layer + 1], 1);
            biasWeightsPerLayer = biasWeightsPerLayer.random(biasWeightsPerLayer.getRowDimension(), biasWeightsPerLayer.getColumnDimension());
            biasWeightsPerLayer = biasWeightsPerLayer.times(2);
            minus(biasWeightsPerLayer, 1);

            weights.put(layer, weightsPerLayer);
            biasWeights.put(layer, biasWeightsPerLayer);
        }
    }

    public void setGradientClipping(boolean gradientClipping) {
        this.gradientClipping = gradientClipping;
    }

    public void setNoTransfer() {
        noTransfer = true;
    }

    public void passForward(double [] input) {
        Matrix inputVector = new Matrix(input, input.length);

        outputs.put(0, inputVector);

        for(int layer = 1; layer <= weights.values().size(); layer++) {

            inputVector = weights.get(layer - 1).times(inputVector);
            inputVector = inputVector.plus(biasWeights.get(layer - 1));

            boolean hidden = layer < weights.values().size();

            Matrix outputVector = transfer(inputVector, hidden);

            outputs.put(layer, outputVector);

            transferDerivertives.put(layer, get2Dim(transferDerivative(outputVector, hidden)));

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

        if(hidden && leakyRelu) {
            for(int row=0; row < vector.getRowDimension(); row++) {
                if(vector.get(row, 0) > 0) {
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

        if(hidden && leakyRelu) {
            for (int kol = 0; kol < vector.getColumnDimension(); kol++) {
                for (int row = 0; row < vector.getRowDimension(); row++) {
                    if(vector.get(row, kol) > 0) {
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

            Matrix theta = null;

            for(int layer=weights.values().size(); layer > 0; layer--) {
                if(theta == null) {
                    theta = transferDerivertives.get(layer).times(errorDeriv).transpose();
                } else {
                    theta = theta.times(weights.get(layer).times(transferDerivertives.get(layer)));
                }
                gradientsPerLayer.put(layer, outputs.get(layer - 1).times(theta).transpose());
                biasGradientsPerLayer.put(layer, theta.transpose());
            }

            for(int layer=weights.values().size(); layer > 0; layer--) {
                if(leakyRelu && gradientClipping) {
                    gradientClipping(gradientsPerLayer.get(layer));
                    gradientClipping(biasGradientsPerLayer.get(layer));
                }
                weights.put(layer - 1, weights.get(layer - 1).minus(gradientsPerLayer.get(layer).times(learningConstant)));
                biasWeights.put(layer - 1, biasWeights.get(layer - 1).minus(biasGradientsPerLayer.get(layer).times(learningConstant)));
            }

            iterations++;

            if(maxIterations > 0 && iterations >= maxIterations) {
                String s = String.format("Max iterations exceeded for classifier %s.", name);
                System.out.println(s);
                return -1;
            }
        }

        return iterations;
    }

    public int learn(double [] inputs, double [] targets, double errorLimit) throws Exception {
        return learn(inputs, targets, errorLimit, 0);
    }

    private void gradientClipping(Matrix gradients) {

        boolean clip = false;

        for (int row = 0; !clip && row < gradients.getRowDimension(); row++) {
            for (int col = 0; !clip && col < gradients.getColumnDimension(); col++) {
                if (gradients.get(row, col) >= GRADIENT_CLIPPING_TRESHOLD) {
                    clip = true;
                    System.out.println(String.format("Gradient %.2f is too big.", gradients.get(row, col)));
                }
            }
        }

        if(!clip) {
            return;
        }

        for (int row = 0; row < gradients.getRowDimension(); row++) {
            for (int col = 0; col < gradients.getColumnDimension(); col++) {
                gradients.set(row, col, (gradients.get(row, col) * GRADIENT_CLIPPING_TRESHOLD) / getL2Norm(gradients));
            }
        }
    }

    Matrix getGradients(int layer) {
        return gradientsPerLayer.get(layer);
    }

    Matrix getBiasGradients(int layer) {
        return biasGradientsPerLayer.get(layer);
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

    public Matrix getBiasWeights(int layer) {
        return biasWeights.get(layer - 1);
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
}
