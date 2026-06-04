package com.tinylpr.config;

import com.tinylpr.core.LprEngine;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * LPR 引擎配置 — 启动时加载 ONNX 模型
 */
@Configuration
public class LprConfig {

    @Value("${lpr.detector-path:models/plate_detector.onnx}")
    private String detectorPath;

    @Value("${lpr.recognizer-path:models/plate_recognizer.onnx}")
    private String recognizerPath;

    @Value("${lpr.confidence-threshold:0.5}")
    private float confidenceThreshold;

    @Bean(destroyMethod = "close")
    public LprEngine lprEngine() throws Exception {
        System.out.println("🚗 加载车牌识别模型...");
        System.out.println("   检测器: " + detectorPath);
        System.out.println("   识别器: " + recognizerPath);
        System.out.println("   置信度阈值: " + confidenceThreshold);

        return new LprEngine(detectorPath, recognizerPath, confidenceThreshold);
    }
}