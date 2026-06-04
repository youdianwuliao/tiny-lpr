package com.tinylpr;

import com.tinylpr.core.LprEngine;
import com.tinylpr.core.PlateResult;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * 命令行测试工具
 *
 * 用法: java -cp ... com.tinylpr.LprTest /path/to/car.jpg
 */
public class LprTest {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("用法: java LprTest <图片路径>");
            System.out.println("示例: java LprTest test.jpg");
            return;
        }

        String imagePath = args[0];
        String detectorPath = args.length > 1 ? args[1] : "models/plate_detector.onnx";
        String recognizerPath = args.length > 2 ? args[2] : "models/plate_recognizer.onnx";

        System.out.println("🚗 TinyLPR 测试");
        System.out.println("   图片: " + imagePath);
        System.out.println("   检测器: " + detectorPath);
        System.out.println("   识别器: " + recognizerPath);

        byte[] imageBytes = Files.readAllBytes(Paths.get(imagePath));

        try (LprEngine engine = new LprEngine(detectorPath, recognizerPath, 0.5f)) {
            long start = System.currentTimeMillis();
            List<PlateResult> results = engine.recognize(imageBytes);
            long elapsed = System.currentTimeMillis() - start;

            System.out.println("\n📊 识别结果 (" + elapsed + "ms):");
            if (results.isEmpty()) {
                System.out.println("   未检测到车牌");
            } else {
                for (PlateResult r : results) {
                    System.out.println("   🚗 " + r);
                }
            }
        }
    }
}