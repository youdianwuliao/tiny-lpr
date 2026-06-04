package com.tinylpr.controller;

import com.tinylpr.core.LprEngine;
import com.tinylpr.core.PlateResult;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;

/**
 * 车牌识别 REST API
 *
 * POST /api/lpr/recognize  — 上传图片识别车牌
 * GET  /api/lpr/health     — 健康检查
 */
@RestController
@RequestMapping("/api/lpr")
public class LprController {

    private final LprEngine engine;

    public LprController(LprEngine engine) {
        this.engine = engine;
    }

    /**
     * 识别车牌
     *
     * @param file 上传的图片文件（JPG/PNG/BMP）
     * @return 识别结果
     */
    @PostMapping("/recognize")
    public Map<String, Object> recognize(@RequestParam("file") MultipartFile file) {
        try {
            List<PlateResult> plates = engine.recognize(file.getBytes());

            List<Map<String, Object>> resultList = new ArrayList<>();
            for (PlateResult p : plates) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("plate", p.getPlate());
                item.put("confidence", Math.round(p.getConfidence() * 10000) / 10000.0);
                item.put("bbox", Arrays.asList(
                        p.getBbox()[0], p.getBbox()[1],
                        p.getBbox()[2], p.getBbox()[3]
                ));
                resultList.add(item);
            }

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("success", true);
            response.put("results", resultList);
            response.put("count", resultList.size());
            return response;
        } catch (Exception e) {
            Map<String, Object> error = new LinkedHashMap<>();
            error.put("success", false);
            error.put("error", e.getMessage());
            return error;
        }
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
}