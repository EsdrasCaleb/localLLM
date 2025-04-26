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
        // Test with normal values
        String result = EWrapperMsgGenerator.historicalData(1, "2023-10-01", 100.0, 110.0, 90.0, 105.0, 1000, 10, 102.5, false);
        assertEquals("id=1 date = 2023-10-01 open=100.0 high=110.0 low=90.0 close=105.0 volume=1000 count=10 WAP=102.5 hasGaps=false", result);
        // Test with edge values
        result = EWrapperMsgGenerator.historicalData(2, "2023-10-02", 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, true);
        assertEquals("id=2 date = 2023-10-02 open=0.0 high=0.0 low=0.0 close=0.0 volume=0 count=0 WAP=0.0 hasGaps=true", result);
        // Test with negative values
        result = EWrapperMsgGenerator.historicalData(-1, "2023-10-03", -100.0, -110.0, -90.0, -105.0, -1000, -10, -102.5, true);
        assertEquals("id=-1 date = 2023-10-03 open=-100.0 high=-110.0 low=-90.0 close=-105.0 volume=-1000 count=-10 WAP=-102.5 hasGaps=true", result);
        // Test with special date format
        result = EWrapperMsgGenerator.historicalData(3, "01-10-2023", 150.5, 160.5, 140.5, 155.5, 1500, 15, 152.5, false);
        assertEquals("id=3 date = 01-10-2023 open=150.5 high=160.5 low=140.5 close=155.5 volume=1500 count=15 WAP=152.5 hasGaps=false", result);
    }
}
