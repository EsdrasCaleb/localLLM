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
        int reqId = 1;
        String date = "2023-10-01";
        double open = 100.0;
        double high = 110.0;
        double low = 90.0;
        double close = 105.0;
        int volume = 1000;
        int count = 50;
        double WAP = 102.5;
        boolean hasGaps = false;
        String expected = "id=1 date = 2023-10-01 open=100.0 high=110.0 low=90.0 close=105.0 volume=1000 count=50 WAP=102.5 hasGaps=false";
        String result = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        assertEquals(expected, result);
    }
}
