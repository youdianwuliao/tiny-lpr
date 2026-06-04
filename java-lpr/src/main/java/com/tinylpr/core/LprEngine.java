package com.tinylpr.core;

import ai.onnxruntime.*;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

/**
 * 纯 Java 车牌识别器 — ONNX Runtime 直接推理
 *
 * 不需要 DJL，只需要 onnxruntime 一个依赖
 * 图片预处理用 JDK 自带的 BufferedImage
 */
public class LprEngine implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession detectorSession;
    private final OrtSession recognizerSession;
    private final float confThreshold;
    private final List<String> charList;

    private static final int DET_INPUT_SIZE = 640;
    private static final int REC_INPUT_W = 160;
    private static final int REC_INPUT_H = 48;

    /**
     * @param detectorPath   检测器 ONNX 路径
     * @param recognizerPath 识别器 ONNX 路径
     * @param confThreshold  检测置信度阈值
     */
    public LprEngine(String detectorPath, String recognizerPath, float confThreshold) throws Exception {
        this.confThreshold = confThreshold;
        this.env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        this.detectorSession = env.createSession(detectorPath, opts);
        this.recognizerSession = env.createSession(recognizerPath, opts);

        // 默认字符集（blank + 省份 + 字母 + 数字）
        this.charList = buildDefaultCharList();

        System.out.println("✅ 检测器加载完成: " + detectorPath);
        System.out.println("✅ 识别器加载完成: " + recognizerPath);
    }

    // ==================== 公开 API ====================

    /**
     * 识别图片中的车牌
     *
     * @param imageBytes 图片字节数组（JPG/PNG/BMP）
     * @return 识别结果列表
     */
    public List<PlateResult> recognize(byte[] imageBytes) throws Exception {
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));
        if (img == null) {
            throw new IOException("无法解析图片");
        }
        return recognize(img);
    }

    /**
     * 识别图片中的车牌
     */
    public List<PlateResult> recognize(BufferedImage img) throws Exception {
        int imgW = img.getWidth();
        int imgH = img.getHeight();

        // 1. 检测车牌位置
        List<float[]> detections = detect(img);
        if (detections.isEmpty()) {
            return Collections.emptyList();
        }

        // 2. 逐个识别
        List<PlateResult> results = new ArrayList<>();
        for (float[] det : detections) {
            int x1 = (int) det[0], y1 = (int) det[1];
            int x2 = (int) det[2], y2 = (int) det[3];
            float conf = det[4];

            // 裁剪车牌区域
            x1 = Math.max(0, x1);
            y1 = Math.max(0, y1);
            x2 = Math.min(imgW, x2);
            y2 = Math.min(imgH, y2);

            if (x2 <= x1 || y2 <= y1) continue;

            BufferedImage crop = img.getSubimage(x1, y1, x2 - x1, y2 - y1);
            String plate = recognizePlate(crop);

            results.add(new PlateResult(plate, conf, x1, y1, x2, y2));
        }

        return results;
    }

    // ==================== 检测 ====================

    private List<float[]> detect(BufferedImage img) throws OrtException {
        // 预处理：resize + normalize
        BufferedImage resized = resize(img, DET_INPUT_SIZE, DET_INPUT_SIZE);
        float[] inputData = imageToFloatArray(resized, DET_INPUT_SIZE, DET_INPUT_SIZE);

        OnnxTensor inputTensor = OnnxTensor.createTensor(env,
                new float[][][][]{{{{0}}}}); // placeholder
        // 实际创建 tensor
        long[] shape = {1, 3, DET_INPUT_SIZE, DET_INPUT_SIZE};
        inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        // 推理
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(detectorSession.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result result = detectorSession.run(inputs);
        OnnxTensor output = (OnnxTensor) result.get(0);

        // 解析 YOLOv8 输出: (1, 84, 8400)
        float[][][] raw = (float[][][]) output.getValue();
        float[][] dets = raw[0]; // (84, 8400)

        return parseDetections(dets, img.getWidth(), img.getHeight());
    }

    private List<float[]> parseDetections(float[][] output, int imgW, int imgH) {
        // output: (84, 8400) → transpose → (8400, 84)
        int numAnchors = output[0].length; // 8400
        int numFeatures = output.length;   // 84

        float scaleX = (float) imgW / DET_INPUT_SIZE;
        float scaleY = (float) imgH / DET_INPUT_SIZE;

        List<float[]> candidates = new ArrayList<>();

        for (int i = 0; i < numAnchors; i++) {
            // 取最大类别置信度
            float maxConf = 0;
            for (int j = 4; j < numFeatures; j++) {
                if (output[j][i] > maxConf) {
                    maxConf = output[j][i];
                }
            }

            if (maxConf < confThreshold) continue;

            float cx = output[0][i] * scaleX;
            float cy = output[1][i] * scaleY;
            float w = output[2][i] * scaleX;
            float h = output[3][i] * scaleY;

            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;

            candidates.add(new float[]{x1, y1, x2, y2, maxConf});
        }

        // 简单 NMS
        return nms(candidates, 0.45f);
    }

    private List<float[]> nms(List<float[]> boxes, float iouThreshold) {
        // 按置信度降序
        boxes.sort((a, b) -> Float.compare(b[4], a[4]));

        List<float[]> result = new ArrayList<>();
        boolean[] suppressed = new boolean[boxes.size()];

        for (int i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            float[] best = boxes.get(i);
            result.add(best);

            for (int j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                if (iou(best, boxes.get(j)) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    private float iou(float[] a, float[] b) {
        float x1 = Math.max(a[0], b[0]);
        float y1 = Math.max(a[1], b[1]);
        float x2 = Math.min(a[2], b[2]);
        float y2 = Math.min(a[3], b[3]);

        if (x2 <= x1 || y2 <= y1) return 0;

        float inter = (x2 - x1) * (y2 - y1);
        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);

        return inter / (areaA + areaB - inter);
    }

    // ==================== 识别 ====================

    private String recognizePlate(BufferedImage crop) throws OrtException {
        // 预处理：resize + normalize
        BufferedImage resized = resize(crop, REC_INPUT_W, REC_INPUT_H);
        float[] inputData = imageToFloatArray(resized, REC_INPUT_W, REC_INPUT_H);

        long[] shape = {1, 3, REC_INPUT_H, REC_INPUT_W};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(recognizerSession.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result result = recognizerSession.run(inputs);
        OnnxTensor output = (OnnxTensor) result.get(0);

        // 输出: (1, T, num_classes)
        float[][][] raw = (float[][][]) output.getValue();
        float[][] logProbs = raw[0]; // (T, num_classes)

        return ctcDecode(logProbs);
    }

    private String ctcDecode(float[][] logProbs) {
        StringBuilder sb = new StringBuilder();
        int prev = -1;

        for (float[] frame : logProbs) {
            // 找最大概率的字符
            int maxIdx = 0;
            float maxVal = frame[0];
            for (int i = 1; i < frame.length; i++) {
                if (frame[i] > maxVal) {
                    maxVal = frame[i];
                    maxIdx = i;
                }
            }

            // 跳过 blank(0) 和重复
            if (maxIdx != 0 && maxIdx != prev) {
                if (maxIdx < charList.size()) {
                    sb.append(charList.get(maxIdx));
                }
            }
            prev = maxIdx;
        }

        return sb.toString();
    }

    // ==================== 图像预处理 ====================

    private BufferedImage resize(BufferedImage src, int w, int h) {
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = dst.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, w, h, null);
        g.dispose();
        return dst;
    }

    /**
     * BufferedImage → float[] (CHW, RGB, normalized to [0,1])
     */
    private float[] imageToFloatArray(BufferedImage img, int w, int h) {
        float[] data = new float[3 * h * w];
        int[] pixels = img.getRGB(0, 0, w, h, null, 0, w);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pixel = pixels[y * w + x];
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;

                // CHW 格式, RGB 顺序, 归一化到 [0, 1]
                data[0 * h * w + y * w + x] = r / 255.0f;
                data[1 * h * w + y * w + x] = g / 255.0f;
                data[2 * h * w + y * w + x] = b / 255.0f;
            }
        }

        return data;
    }

    // ==================== 字符集 ====================

    private static List<String> buildDefaultCharList() {
        List<String> chars = new ArrayList<>();
        chars.add("-"); // blank

        // 省份简称
        String[] provinces = {"京","津","冀","晋","蒙","辽","吉","黑","沪","苏",
                "浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂",
                "琼","渝","川","贵","云","藏","陕","甘","青","宁","新"};

        // 字母（不含 I、O）
        String letters = "ABCDEFGHJKLMNPQRSTUVWXYZ";

        // 数字
        String digits = "0123456789";

        for (String p : provinces) chars.add(p);
        for (char c : letters.toCharArray()) chars.add(String.valueOf(c));
        for (char c : digits.toCharArray()) chars.add(String.valueOf(c));

        return chars;
    }

    @Override
    public void close() {
        try { detectorSession.close(); } catch (Exception ignored) {}
        try { recognizerSession.close(); } catch (Exception ignored) {}
        try { env.close(); } catch (Exception ignored) {}
    }
}