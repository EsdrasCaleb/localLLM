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
    public void realtimeBarTest() {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        int requestId = 123;
        long timestamp = 1234567890L;
        double open = 10.0;
        double high = 100.0;
        double low = 5.0;
        double close = 20.0;
        long volume = 1000L;
        double wap = 20.0;
        int count = 5;
        // Act
        String result = eWrapperMsgGenerator.realtimeBar(requestId, timestamp, open, high, low, close, volume, wap, count);
        // Assert
        assertEquals("123:1234567890:10.0:100.0:5.0:20.0:1000:20.0:5", result);
    }
}
