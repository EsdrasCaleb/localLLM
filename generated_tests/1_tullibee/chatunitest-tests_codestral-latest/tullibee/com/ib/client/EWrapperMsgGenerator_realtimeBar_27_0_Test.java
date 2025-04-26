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

class EWrapperMsgGenerator_realtimeBar_27_0_Test {

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    void setUp() {
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
    }

    @Test
    void testRealtimeBar() throws Exception {
        int reqId = 1;
        // 2021-10-01 00:00:00
        long time = 1633072800000L;
        double open = 100.0;
        double high = 110.0;
        double low = 90.0;
        double close = 105.0;
        long volume = 1000L;
        double wap = 102.5;
        int count = 50;
        Method realtimeBarMethod = EWrapperMsgGenerator.class.getDeclaredMethod("realtimeBar", int.class, long.class, double.class, double.class, double.class, double.class, long.class, double.class, int.class);
        realtimeBarMethod.setAccessible(true);
        String expected = "1 1633072800000 100.0 110.0 90.0 105.0 1000 102.5 50";
        String result = (String) realtimeBarMethod.invoke(eWrapperMsgGenerator, reqId, time, open, high, low, close, volume, wap, count);
        assertEquals(expected, result);
    }
}
