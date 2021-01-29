package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;

import java.io.IOException;

public class ClassifierSavedModel extends Classifier {

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
}
