package com.tinylpr.core;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 车牌检测器 — 基于 YOLOv8n
 * 使用 DJL 加载 ONNX 模型进行推理
 */
public class PlateDetector implements AutoCloseable {

    private final ZooModel<Image, DetectedObjects> model;
    private final Predictor<Image, DetectedObjects> predictor;
    private final float confidenceThreshold;

    public PlateDetector(String modelPath, float confidenceThreshold) throws IOException, ModelException {
        this.confidenceThreshold = confidenceThreshold;

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optModelPath(Paths.get(modelPath))
                .optEngine("OnnxRuntime")
                .optTranslator(new YoloV8Translator())
                .build();

        this.model = ModelZoo.loadModel(criteria);
        this.predictor = model.newPredictor();
    }

    /**
     * 检测图片中的车牌区域
     *
     * @param image DJL Image 对象
     * @return 检测到的车牌区域列表 [x1, y1, x2, y2, confidence]
     */
    public List<float[]> detect(Image image) throws TranslateException {
        DetectedObjects detections = predictor.predict(image);
        List<float[]> results = new ArrayList<>();

        for (DetectedObjects.DetectedObject obj : detections.items()) {
            if (obj.getProbability() < confidenceThreshold) {
                continue;
            }

            Rectangle rect = obj.getBoundingBox().getBounds();
            results.add(new float[]{
                    (float) rect.getX(),
                    (float) rect.getY(),
                    (float) (rect.getX() + rect.getWidth()),
                    (float) (rect.getY() + rect.getHeight()),
                    (float) obj.getProbability()
            });
        }

        return results;
    }

    @Override
    public void close() {
        if (predictor != null) predictor.close();
        if (model != null) model.close();
    }
}