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

public class EWrapperMsgGenerator_orderStatus_6_0_Test {

    @Test
    public void testOrderStatus() throws Exception {
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        Method method = EWrapperMsgGenerator.class.getDeclaredMethod("orderStatus", int.class, String.class, int.class, int.class, double.class, int.class, int.class, double.class, int.class, String.class);
        method.setAccessible(true);
        String result = (String) method.invoke(generator, 1, "FILLED", 10, 0, 100.0, 1001, 1002, 100.0, 1003, "REPLACED");
        assertEquals("order status: orderId=1 clientId=1001 permId=1002 status=FILLED filled=10 remaining=0 avgFillPrice=100.0 lastFillPrice=100.0 parent Id=1002 whyHeld=REPLACED", result);
    }
}
