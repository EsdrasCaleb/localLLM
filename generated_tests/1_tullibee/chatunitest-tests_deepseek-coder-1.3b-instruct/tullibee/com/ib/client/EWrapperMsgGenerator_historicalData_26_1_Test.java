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

public class EWrapperMsgGenerator_historicalData_26_1_Test {

    @Test
    public void testHistoricalData() {
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        int reqId = 1;
        String date = "2022-01-01";
        double open = 100.0;
        double high = 101.0;
        double low = 99.0;
        double close = 102.0;
        int volume = 1000;
        int count = 1;
        double WAP = 101.0;
        boolean hasGaps = false;
        String expected = "id=1 date = " + date + " open=100.0 high=101.0 low=99.0 close=102.0 volume=1000 count=1 WAP=101.0 hasGaps=false";
        assertEquals(expected, eWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps));
    }
}
