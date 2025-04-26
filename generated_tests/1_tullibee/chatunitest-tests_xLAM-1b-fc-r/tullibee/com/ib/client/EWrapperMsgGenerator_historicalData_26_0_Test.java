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
        // Mock the arguments
        int reqId = 123;
        String date = "2020-01-01";
        double open = 10.0;
        double high = 10.0;
        double low = 10.0;
        double close = 10.0;
        int volume = 100;
        int count = 10;
        double WAP = 10.0;
        boolean hasGaps = true;
        // Call the method with the mocked arguments
        String result = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        // Assert the result
        assertEquals("id=123 date = " + date + " open=10.0 high=10.0 low=10.0 close=10.0 volume=100 count=10 WAP=10.0 hasGaps=true", result);
    }
}
