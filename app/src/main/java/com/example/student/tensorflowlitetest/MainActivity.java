package com.example.student.tensorflowlitetest;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;


import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitImage;

import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private ImageClassifier imageClassifier;
    private static final String MODEL_PATH = "mobilenet_quant_v1_224.tflite";
    private static final String LABEL_PATH = "quantlabels.txt";
    private static final int INPUT_SIZE = 224;

    private CameraView cameraView;
    private TextView resultTextView;
    private Button captureButton;
    private float[][][][] imgData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraView = findViewById(R.id.camera);
        resultTextView = findViewById(R.id.tvClassificationResult);
        captureButton = findViewById(R.id.btnCapture);
        try {
            loadClassifier();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.i("Status","Loaded!");
        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {}
            @Override
            public void onError(CameraKitError cameraKitError) {}
            @Override
            public void onImage(CameraKitImage cameraKitImage) {
                Bitmap bitmap = cameraKitImage.getBitmap();
                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
                final List<Recognition> results = imageClassifier.recognizeImage(bitmap);
                resultTextView.setText("Prob: " + results.toString());
                Log.i("Prob:", results.toString());
            }
            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {}
        });
        captureButton.setOnClickListener(v -> cameraView.captureImage());
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    private void loadClassifier() throws IOException {
        imageClassifier = new ImageClassifier(this, MODEL_PATH, LABEL_PATH, ImageClassifier.Type.QUANT);
    }
}


