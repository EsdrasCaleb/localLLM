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

class EWrapperMsgGenerator_realtimeBar_27_1_Test {

    @Test
    public void testRealtimeBar() {
        // Arrange
        int reqId = 1;
        long time = 1630480000;
        double open = 100.0;
        double high = 102.0;
        double low = 98.0;
        double close = 101.0;
        long volume = 1000000;
        double wap = 100.0;
        int count = 5;
        // Act
        String result = EWrapperMsgGenerator.realtimeBar(reqId, time, open, high, low, close, volume, wap, count);
        // Assert
        assertEquals("bar1", result);
    }
}
