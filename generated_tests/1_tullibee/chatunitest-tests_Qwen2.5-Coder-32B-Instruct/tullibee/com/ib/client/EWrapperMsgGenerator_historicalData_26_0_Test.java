package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_historicalData_26_0_Test {

    @Test
    public void testHistoricalData() {
        // Test data
        int reqId = 12345;
        String date = "20231001";
        double open = 150.75;
        double high = 155.25;
        double low = 149.50;
        double close = 153.00;
        int volume = 1000000;
        int count = 500;
        double WAP = 152.75;
        boolean hasGaps = true;
        // Expected result
        String expected = "id=" + reqId + " date = " + date + " open=" + open + " high=" + high + " low=" + low + " close=" + close + " volume=" + volume + " count=" + count + " WAP=" + WAP + " hasGaps=" + hasGaps;
        // Actual result
        String actual = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        // Assertion
        assertEquals(expected, actual);
    }

    @Test
    public void testHistoricalDataWithNoGaps() {
        // Test data
        int reqId = 67890;
        String date = "20231002";
        double open = 145.00;
        double high = 147.50;
        double low = 144.00;
        double close = 146.25;
        int volume = 800000;
        int count = 400;
        double WAP = 146.00;
        boolean hasGaps = false;
        // Expected result
        String expected = "id=" + reqId + " date = " + date + " open=" + open + " high=" + high + " low=" + low + " close=" + close + " volume=" + volume + " count=" + count + " WAP=" + WAP + " hasGaps=" + hasGaps;
        // Actual result
        String actual = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        // Assertion
        assertEquals(expected, actual);
    }
}
