package com.ib.client;

import java.lang.reflect.Method;
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

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    public void setUp() {
        eWrapperMsgGenerator = Mockito.spy(new EWrapperMsgGenerator());
    }

    @Test
    public void testRealtimeBar() throws Exception {
        // Arrange
        int reqId = 1;
        long time = 1633072800L;
        double open = 150.0;
        double high = 155.0;
        double low = 145.0;
        double close = 152.0;
        long volume = 1000L;
        double wap = 151.0;
        int count = 50;
        String expectedOutput = "reqId=1, time=1633072800, open=150.0, high=155.0, low=145.0, close=152.0, volume=1000, wap=151.0, count=50";
        // Access the private method using reflection
        Method realtimeBarMethod = EWrapperMsgGenerator.class.getDeclaredMethod("realtimeBar", int.class, long.class, double.class, double.class, double.class, double.class, long.class, double.class, int.class);
        realtimeBarMethod.setAccessible(true);
        // Act
        String result = (String) realtimeBarMethod.invoke(eWrapperMsgGenerator, reqId, time, open, high, low, close, volume, wap, count);
        // Assert
        assertEquals(expectedOutput, result);
    }
}
