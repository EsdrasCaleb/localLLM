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

class EWrapperMsgGenerator_historicalData_26_2_Test {

    @Test
    void testHistoricalData() {
        // Arrange
        Integer reqId = 123;
        String date = "2023-04-01";
        double open = 10.0;
        double high = 12.0;
        double low = 8.0;
        double close = 11.0;
        int volume = 1000;
        int count = 500;
        double WAP = 10.5;
        boolean hasGaps = true;
        // Act
        String result = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        // Assert
        assertEquals("id=123 date = 2023-04-01 open=10.0 high=12.0 low=8.0 close=11.0 volume=1000 count=500 WAP=10.5 hasGaps=true", result);
    }
}
