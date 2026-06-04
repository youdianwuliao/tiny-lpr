package com.tinylpr;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * TinyLPR — 纯 Java 车牌识别服务
 *
 * 启动后访问 http://localhost:8080
 * API: POST /api/recognize?file=图片
 */
@SpringBootApplication
public class LprApplication {

    public static void main(String[] args) {
        System.out.println("🚗 TinyLPR Java 服务启动中...");
        SpringApplication.run(LprApplication.class, args);
        System.out.println("✅ 服务已启动: http://localhost:8080");
    }
}