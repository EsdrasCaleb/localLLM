package com.ib.client;

import java.lang.reflect.InvocationTargetException;
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

class EWrapperMsgGenerator_orderStatus_6_0_Test {

    @Test
    void testOrderStatus_allFieldsPopulated() {
        String result = EWrapperMsgGenerator.orderStatus(123, "Filled", 100, 0, 10.50, 456, 0, 10.50, 789, "");
        assertEquals("order status: orderId=123 clientId=789 permId=456 status=Filled filled=100 remaining=0 avgFillPrice=10.5 lastFillPrice=10.5 parent Id=0 whyHeld=", result);
    }

    @Test
    void testOrderStatus_partiallyFilled() {
        String result = EWrapperMsgGenerator.orderStatus(456, "PartiallyFilled", 50, 50, 10.00, 789, 123, 10.25, 101, "Long trade");
        assertEquals("order status: orderId=456 clientId=101 permId=789 status=PartiallyFilled filled=50 remaining=50 avgFillPrice=10.0 lastFillPrice=10.25 parent Id=123 whyHeld=Long trade", result);
    }

    @Test
    void testOrderStatus_cancelled() {
        String result = EWrapperMsgGenerator.orderStatus(789, "Cancelled", 0, 100, 0.0, 101, 0, 0.0, 123, "Insufficient Funds");
        assertEquals("order status: orderId=789 clientId=123 permId=101 status=Cancelled filled=0 remaining=100 avgFillPrice=0.0 lastFillPrice=0.0 parent Id=0 whyHeld=Insufficient Funds", result);
    }

    @Test
    void testOrderStatus_zeroValues() {
        String result = EWrapperMsgGenerator.orderStatus(0, "Filled", 0, 0, 0.0, 0, 0, 0.0, 0, null);
        assertEquals("order status: orderId=0 clientId=0 permId=0 status=Filled filled=0 remaining=0 avgFillPrice=0.0 lastFillPrice=0.0 parent Id=0 whyHeld=null", result);
    }

    @Test
    void testOrderStatus_negativeValues() {
        String result = EWrapperMsgGenerator.orderStatus(-123, "Filled", -100, -50, -10.50, -456, -100, -10.50, -789, "test");
        assertEquals("order status: orderId=-123 clientId=-789 permId=-456 status=Filled filled=-100 remaining=-50 avgFillPrice=-10.5 lastFillPrice=-10.5 parent Id=-100 whyHeld=test", result);
    }
}
