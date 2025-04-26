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

public class EWrapperMsgGenerator_realtimeBar_27_0_Test {

    @Test
    public void testRealtimeBar() throws Exception {
        // Arrange
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        int reqId = 1;
        // Example timestamp
        long time = 1622548800L;
        double open = 100.0;
        double high = 105.0;
        double low = 99.0;
        double close = 102.0;
        long volume = 1000L;
        double wap = 101.0;
        int count = 10;
        // Act
        String result = (String) EWrapperMsgGenerator.class.getDeclaredMethod("realtimeBar", int.class, long.class, double.class, double.class, double.class, double.class, long.class, double.class, int.class).invoke(generator, reqId, time, open, high, low, close, volume, wap, count);
        // Assert
        String expected = "Request ID: 1, Time: 1622548800, Open: 100.0, High: 105.0, Low: 99.0, Close: 102.0, Volume: 1000, WAP: 101.0, Count: 10";
        assertEquals(expected, result);
    }
}
