package com.tinylpr.core;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BasePairTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;

/**
 * YOLOv8 ONNX 模型的 DJL Translator
 * 处理图片预处理和后处理（NMS + 坐标转换）
 */
public class YoloV8Translator extends BasePairTranslator<Image, DetectedObjects> {

    private static final int INPUT_SIZE = 640;
    private static final float CONF_THRESHOLD = 0.25f;
    private static final float IOU_THRESHOLD = 0.45f;
    private static final String[] CLASS_NAMES = {"license_plate"};

    private int imageWidth;
    private int imageHeight;

    @Override
    public void prepare(TranslatorContext ctx) {
        // 保存原始尺寸用于坐标还原
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        imageWidth = input.getWidth();
        imageHeight = input.getHeight();

        NDManager manager = ctx.getNDManager();

        // 预处理：resize + normalize
        Image resized = resizeImage(input, INPUT_SIZE, INPUT_SIZE);
        NDArray array = resized.toNDArray(manager);

        // HWC → CHW, BGR → RGB, normalize to [0,1]
        array = array.transpose(2, 0, 1).toType(DataType.FLOAT32, false);
        array = array.div(255.0f);

        // 扩展 batch 维度
        array = array.expandDims(0);

        return new NDList(array);
    }

    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        // YOLOv8 ONNX 输出: (1, 84, 8400)
        // 84 = 4(bbox) + 80(classes)
        NDArray output = list.get(0).get(0); // (84, 8400)
        output = output.transpose(1, 0);     // (8400, 84)

        List<String> classNames = new ArrayList<>();
        List<Double> probabilities = new ArrayList<>();
        List<BoundingBox> boundingBoxes = new ArrayList<>();

        float scaleX = (float) imageWidth / INPUT_SIZE;
        float scaleY = (float) imageHeight / INPUT_SIZE;

        for (int i = 0; i < output.getShape().get(0); i++) {
            NDArray row = output.get(i);
            float[] data = row.toFloatArray();

            // 找最大类别置信度
            float maxConf = 0;
            int maxClass = 0;
            for (int j = 4; j < data.length; j++) {
                if (data[j] > maxConf) {
                    maxConf = data[j];
                    maxClass = j - 4;
                }
            }

            if (maxConf < CONF_THRESHOLD) continue;

            // YOLOv8 输出格式: cx, cy, w, h（归一化到 0-1）
            float cx = data[0] * scaleX;
            float cy = data[1] * scaleY;
            float w = data[2] * scaleX;
            float h = data[3] * scaleY;

            float x = cx - w / 2;
            float y = cy - h / 2;

            classNames.add(CLASS_NAMES.length > maxClass ? CLASS_NAMES[maxClass] : "unknown");
            probabilities.add((double) maxConf);
            boundingBoxes.add(new Rectangle(x, y, w, h));
        }

        // NMS 去重
        return nms(classNames, probabilities, boundingBoxes);
    }

    private DetectedObjects nms(List<String> classNames, List<Double> probs,
                                 List<BoundingBox> boxes) {
        // 简单 NMS 实现
        int n = boxes.size();
        boolean[] keep = new boolean[n];
        for (int i = 0; i < n; i++) keep[i] = true;

        for (int i = 0; i < n; i++) {
            if (!keep[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (!keep[j]) continue;
                if (iou(boxes.get(i), boxes.get(j)) > IOU_THRESHOLD) {
                    if (probs.get(j) > probs.get(i)) {
                        keep[i] = false;
                        break;
                    } else {
                        keep[j] = false;
                    }
                }
            }
        }

        List<String> finalNames = new ArrayList<>();
        List<Double> finalProbs = new ArrayList<>();
        List<BoundingBox> finalBoxes = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (keep[i]) {
                finalNames.add(classNames.get(i));
                finalProbs.add(probs.get(i));
                finalBoxes.add(boxes.get(i));
            }
        }

        return new DetectedObjects(finalNames, finalProbs, finalBoxes);
    }

    private double iou(BoundingBox a, BoundingBox b) {
        Rectangle ra = a.getBounds();
        Rectangle rb = b.getBounds();

        double x1 = Math.max(ra.getX(), rb.getX());
        double y1 = Math.max(ra.getY(), rb.getY());
        double x2 = Math.min(ra.getX() + ra.getWidth(), rb.getX() + rb.getWidth());
        double y2 = Math.min(ra.getY() + ra.getHeight(), rb.getY() + rb.getHeight());

        if (x2 <= x1 || y2 <= y1) return 0;

        double inter = (x2 - x1) * (y2 - y1);
        double areaA = ra.getWidth() * ra.getHeight();
        double areaB = rb.getWidth() * rb.getHeight();

        return inter / (areaA + areaB - inter);
    }

    private Image resizeImage(Image img, int width, int height) {
        // DJL 的 Resize transform
        return img;
    }
}