package com.tinylpr.controller;

import com.tinylpr.core.LprEngine;
import com.tinylpr.core.PlateResult;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * 车牌识别 REST API（支持同步 + 异步）
 */
@RestController
@RequestMapping("/api/lpr")
public class LprController {

    private final LprEngine engine;

    public LprController(LprEngine engine) {
        this.engine = engine;
    }

    /**
     * 同步识别
     * POST /api/lpr/recognize
     */
    @PostMapping("/recognize")
    public Map<String, Object> recognize(@RequestParam("file") MultipartFile file) {
        try {
            long start = System.currentTimeMillis();
            List<PlateResult> plates = engine.recognize(file.getBytes());
            long elapsed = System.currentTimeMillis() - start;

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("success", true);
            response.put("elapsed_ms", elapsed);
            response.put("count", plates.size());
            response.put("results", formatResults(plates));
            return response;
        } catch (Exception e) {
            return errorResponse(e.getMessage());
        }
    }

    /**
     * 异步识别（高并发场景）
     * POST /api/lpr/recognize-async
     */
    @PostMapping("/recognize-async")
    public CompletableFuture<Map<String, Object>> recognizeAsync(
            @RequestParam("file") MultipartFile file) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                List<PlateResult> plates = engine.recognize(file.getBytes());
                Map<String, Object> response = new LinkedHashMap<>();
                response.put("success", true);
                response.put("count", plates.size());
                response.put("results", formatResults(plates));
                return response;
            } catch (Exception e) {
                return errorResponse(e.getMessage());
            }
        });
    }

    /**
     * 批量识别
     * POST /api/lpr/recognize-batch
     */
    @PostMapping("/recognize-batch")
    public Map<String, Object> recognizeBatch(@RequestParam("files") List<MultipartFile> files) {
        Map<String, Object> response = new LinkedHashMap<>();
        List<Map<String, Object>> batchResults = new ArrayList<>();
        long totalStart = System.currentTimeMillis();

        for (MultipartFile file : files) {
            try {
                List<PlateResult> plates = engine.recognize(file.getBytes());
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("filename", file.getOriginalFilename());
                item.put("results", formatResults(plates));
                batchResults.add(item);
            } catch (Exception e) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("filename", file.getOriginalFilename());
                item.put("error", e.getMessage());
                batchResults.add(item);
            }
        }

        response.put("success", true);
        response.put("total_ms", System.currentTimeMillis() - totalStart);
        response.put("files", batchResults);
        return response;
    }

    /** 健康检查 */
    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("status", "ok");
        status.put("model", "loaded");
        status.put("version", "1.0.0");
        return status;
    }

    private List<Map<String, Object>> formatResults(List<PlateResult> plates) {
        List<Map<String, Object>> list = new ArrayList<>();
        for (PlateResult p : plates) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("plate", p.getPlate());
            item.put("confidence", Math.round(p.getConfidence() * 10000) / 10000.0);
            item.put("bbox", Arrays.asList(
                    p.getBbox()[0], p.getBbox()[1],
                    p.getBbox()[2], p.getBbox()[3]));
            list.add(item);
        }
        return list;
    }

    private Map<String, Object> errorResponse(String msg) {
        Map<String, Object> err = new LinkedHashMap<>();
        err.put("success", false);
        err.put("error", msg);
        return err;
    }
}