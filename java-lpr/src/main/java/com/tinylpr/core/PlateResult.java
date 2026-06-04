package com.tinylpr.core;

import java.util.List;

/**
 * 车牌识别结果
 */
public class PlateResult {

    /** 车牌号，如 "京A12345" */
    private String plate;

    /** 置信度 0-1 */
    private double confidence;

    /** 边界框 [x1, y1, x2, y2] */
    private int[] bbox;

    public PlateResult() {}

    public PlateResult(String plate, double confidence, int x1, int y1, int x2, int y2) {
        this.plate = plate;
        this.confidence = confidence;
        this.bbox = new int[]{x1, y1, x2, y2};
    }

    public String getPlate() { return plate; }
    public void setPlate(String plate) { this.plate = plate; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }

    public int[] getBbox() { return bbox; }
    public void setBbox(int[] bbox) { this.bbox = bbox; }

    @Override
    public String toString() {
        return String.format("PlateResult{plate='%s', confidence=%.2f, bbox=[%d,%d,%d,%d]}",
                plate, confidence, bbox[0], bbox[1], bbox[2], bbox[3]);
    }
}