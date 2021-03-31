package com.otaliastudios.cameraview.demo.tflite;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.view.Surface;

import androidx.annotation.NonNull;

import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.demo.CameraActivity;
import com.otaliastudios.cameraview.demo.tflite.Classifier.Recognition;
import com.otaliastudios.cameraview.engine.CameraEngine;
import com.otaliastudios.cameraview.frame.Frame;
import com.otaliastudios.cameraview.frame.FrameProcessor;

import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.List;

/**
 * Description:
 * <p> CameraActivity add TensorFlow flower inference feature;
 * Create by zhaojialiang02  2021/3/30 4:15 PM
 */
public class TensorFlowCameraActivity extends CameraActivity {
    private static final String TAG = "TFActivity";

    private Classifier classifier;
    private int sensorOrientation;
    private long lastProcessingTimeMs;

    private Handler handler;
    private HandlerThread handlerThread;

    /** Input image size of the model along x axis. */
    private int imageSizeX;
    /** Input image size of the model along y axis. */
    private int imageSizeY;
    private FrameProcessor processor = new FrameProcessor() {
        @Override
        public void process(@NonNull Frame frame) {
//           Bitmap bitmap =  getBitmap(frame);
//           Log.i(TAG, "Bitmap is null?" + (bitmap == null ? "isNull" : bitmap));
            CameraEngine engine = getCamera().getCameraEngin();
            processImage(engine.getRgbFrameBitmap());
            engine.readyForNextImage();

        }
    };

    private int getNumThreads() {
        return 4;
    }

    private Classifier.Device getDevice() {
        return Classifier.Device.GPU;
    }

    @Override
    public void configCamera(@NotNull CameraView camera) {
        super.configCamera(camera);
        camera.addFrameProcessor(processor);
    }

    @Override
    public synchronized void onResume() {
        Log.d(TAG, "onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }


    @Override
    public synchronized void onPause() {
        Log.d(TAG, "onPause " + this);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            Log.e(TAG, "Exception!");
        }

        super.onPause();
    }

    /**
     *  Camera2: onResume onSurfaceTextureAvailable
     *  public void onPreviewSizeChosen(final Size size, final int rotation) {
     *      CameraActivity.this.onPreviewSizeChosen(size, rotation);
     *
     *  Camera1:
     *  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
     *      onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
     * @param rotation
     */
    @Override
    public void onPreviewSizeChange(int width, int height) {
        recreateClassifier(getDevice(), getNumThreads());
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

//    /**
//     * Device 配置更改再调用，目前不需要
//     */
//    public void onInferenceConfigurationChanged() {
//        if (rgbFrameBitmap == null) {
//            // Defer creation until we're getting camera frames.
//            return;
//        }
//        final Classifier.Device device = getDevice();
//        final int numThreads = getNumThreads();
//        runInBackground(new Runnable() {
//            @Override
//            public void run() {
//                TensorFlowCameraActivity.this.recreateClassifier(device, numThreads);
//            }
//        });
//    }

    public void processImage(final Bitmap bitmap) {
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if (classifier != null) {
                            final long startTime = SystemClock.uptimeMillis();
                            final List<Classifier.Recognition> results =
                                    classifier.recognizeImage(bitmap, sensorOrientation);
//                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            Log.d(TAG, "Detect: " + results);

                            showResultsInBottomSheet(results);

                        }
                    }
                });
    }

    private void showResultsInBottomSheet(List<Recognition> results) {
        if (results != null && results.size() >= 3) {
            StringBuilder builder = new StringBuilder();

            Recognition recognition = results.get(0);
            if (recognition != null) {
                if (recognition.getTitle() != null) {
                    builder.append(recognition.getTitle());
                }
                if (recognition.getConfidence() != null) {
                    builder.append(String.format("%.2f", (100 * recognition.getConfidence())) + "%");
                }
            }
            builder.append(" ____ ");

            Recognition recognition1 = results.get(1);
            if (recognition1 != null) {
                if (recognition1.getTitle() != null) {
                    builder.append(recognition1.getTitle());
                }
                if (recognition1.getConfidence() != null) {
                    builder.append(
                            String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
                }
            }
            builder.append(" ____ ");

            Recognition recognition2 = results.get(2);
            if (recognition2 != null) {
                if (recognition2.getTitle() != null) {
                    builder.append(recognition2.getTitle());
                }
                if (recognition2.getConfidence() != null) {
                    builder.append(
                            String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
                }
            }

            Log.i(TAG, "Result: " + builder.toString());
        }

    }

    private void recreateClassifier(Classifier.Device device, int numThreads) {
        if (classifier != null) {
            Log.d(TAG, "Closing classifier.");
            classifier.close();
            classifier = null;
        }
        try {
            Log.d(TAG, String.format("Creating classifier (device=%s, numThreads=%d)", device, numThreads));
            classifier = Classifier.create(this, device, numThreads);
        } catch (IOException e) {
            Log.e(TAG, "Failed to create classifier.");
        }

        // Updates the input image size.
        imageSizeX = classifier.getImageSizeX();
        imageSizeY = classifier.getImageSizeY();
    }

    private synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

}
