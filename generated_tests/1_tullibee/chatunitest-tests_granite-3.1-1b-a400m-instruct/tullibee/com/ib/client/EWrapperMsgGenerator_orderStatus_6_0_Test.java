package com.ib.client;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
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
    public void testOrderStatus() {
        // Mocking the order status method
        String mockOrderStatus = "order status: orderId=12345, clientId=12345, permId=12345, status=FILLED, filled=100, remaining=250, avgFillPrice=100.0, lastFillPrice=100.0, parent Id=12345, whyHeld=Hold for Price Improvement";
        String expectedOrderStatus = "order status: orderId=12345, clientId=12345, permId=12345, status=FILLED, filled=100, remaining=250, avgFillPrice=100.0, lastFillPrice=100.0, parent Id=12345, whyHeld=Hold for Price Improvement";
        // Mocking the order status method with the mock data
        EWrapperMsgGenerator.orderStatus(12345, "FILLED", 100, 250, 100.0, 12345, 12345, 100.0, 12345, "Hold for Price Improvement");
        // Asserting the actual order status matches the expected order status
        assertEquals(mockOrderStatus, EWrapperMsgGenerator.orderStatus(12345, "FILLED", 100, 250, 100.0, 12345, 12345, 100.0, 12345, "Hold for Price Improvement"));
    }
}
