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
    public void testFundamentalData() {
        int reqId = 1;
        String data = "Test data";
        String expected = "id  = 1 len = 9\nTest data";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }

    @Test
    public void testFundamentalDataWithEmptyData() {
        int reqId = 2;
        String data = "";
        String expected = "id  = 2 len = 0\n";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }

    @Test
    public void testFundamentalDataWithNullData() {
        int reqId = 3;
        String data = null;
        String expected = "id  = 3 len = 4\nnull";
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, result);
    }
}
