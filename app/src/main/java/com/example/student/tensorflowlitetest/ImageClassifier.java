package com.example.student.tensorflowlitetest;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class ImageClassifier {
    public enum Type {FLOAT, QUANT};
    private Type modelType;
    private Interpreter tflite;
    private List<String> labelList;
    private float[][][][] imgDataFloat;
    private ByteBuffer imgDataQuant;
    private static final int INPUT_SIZE = 224;
    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    public ImageClassifier(Activity activity, String MODEL_PATH, String LABEL_PATH, Type modelType) throws IOException { ;
        this.modelType = modelType;
        this.tflite = new Interpreter(loadModelFile(activity, MODEL_PATH));
        this.labelList = loadLabelList(activity, LABEL_PATH);
    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_PATH) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(Activity activity, String LABEL_PATH) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    public List<Recognition> recognizeImage(Bitmap bitmap) {
        if (modelType == Type.QUANT) {
            convertBitmapToByteBuffer(bitmap);
            byte[][] result = new byte[1][labelList.size()];
            tflite.run(imgDataQuant, result);
            return getSortedQuantResult(result);
        } else if (modelType == Type.FLOAT) {
            convertBitmapToFloatArray(bitmap);
            float[][] result = new float[1][labelList.size()];
            tflite.run(imgDataFloat, result);
            return getSortedFloatResult(result);
        }
        return null;
    }

    private void convertBitmapToFloatArray(Bitmap bitmap) {
        imgDataFloat = new float[BATCH_SIZE][INPUT_SIZE][INPUT_SIZE][PIXEL_SIZE];
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                imgDataFloat[0][i][j][0] = (float) ((val >> 16) & 0xFF);
                imgDataFloat[0][i][j][1] = (float) ((val >> 8) & 0xFF);
                imgDataFloat[0][i][j][2] = (float) (val & 0xFF);
            }
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        imgDataQuant = ByteBuffer.allocateDirect(BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        imgDataQuant.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                imgDataQuant.put((byte) ((val >> 16) & 0xFF));
                imgDataQuant.put((byte) ((val >> 8) & 0xFF));
                imgDataQuant.put((byte) (val & 0xFF));
            }
        }
        return imgDataQuant;
    }

    private List<Recognition> getSortedFloatResult(float[][] labelProbArray) {
        Log.i("Prob:", labelProbArray.toString());
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        (lhs, rhs) -> Float.compare(rhs.getConfidence(), lhs.getConfidence()));

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i]);
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    private List<Recognition> getSortedQuantResult(byte[][] labelProbArray) {

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

 }
