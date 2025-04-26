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
    public void testFundamentalDataWithEmptyString() {
        int reqId = 1;
        String data = "";
        String expected = "id  = 1 len = 0\n";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }

    @Test
    public void testFundamentalDataWithNonEmptyString() {
        int reqId = 2;
        String data = "Sample Data";
        String expected = "id  = 2 len = 11\nSample Data";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }

    @Test
    public void testFundamentalDataWithNullString() {
        int reqId = 3;
        String data = null;
        Exception exception = assertThrows(NullPointerException.class, () -> {
            EWrapperMsgGenerator.fundamentalData(reqId, data);
        });
        assertNotNull(exception);
    }

    @Test
    public void testFundamentalDataWithLargeString() {
        int reqId = 4;
        String data = "a".repeat(1000);
        String expected = "id  = 4 len = 1000\n" + data;
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }
}
