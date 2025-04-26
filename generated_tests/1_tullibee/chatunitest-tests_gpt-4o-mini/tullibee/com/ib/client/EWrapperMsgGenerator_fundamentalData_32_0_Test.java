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

public class EWrapperMsgGenerator_fundamentalData_32_0_Test {

    @Test
    public void testFundamentalData_ValidInput() {
        int reqId = 1;
        String data = "Sample Data";
        String expected = "id  = 1 len = 12\nSample Data";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }

    @Test
    public void testFundamentalData_EmptyData() {
        int reqId = 2;
        String data = "";
        String expected = "id  = 2 len = 0\n";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }

    @Test
    public void testFundamentalData_NegativeReqId() {
        int reqId = -1;
        String data = "Negative ID";
        String expected = "id  = -1 len = 12\nNegative ID";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }

    @Test
    public void testFundamentalData_ZeroReqId() {
        int reqId = 0;
        String data = "Zero ID";
        String expected = "id  = 0 len = 8\nZero ID";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }
}
