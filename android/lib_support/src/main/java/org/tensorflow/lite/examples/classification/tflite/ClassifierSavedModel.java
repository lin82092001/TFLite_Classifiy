package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

public class ClassifierSavedModel extends Classifier {

    /**
     * Float MobileNet requires additional normalization of the used input.
     */
    private static final float IMAGE_MEAN = 127.0f;

    private static final float IMAGE_STD = 128.0f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    /**
     * Initializes a {@code Classifier}.
     *
     * @param activity
     * @param device
     * @param numThreads
     */
    protected ClassifierSavedModel(Activity activity, Device device, int numThreads) throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() {
        return "saved_model.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "label_saved.txt";
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}
