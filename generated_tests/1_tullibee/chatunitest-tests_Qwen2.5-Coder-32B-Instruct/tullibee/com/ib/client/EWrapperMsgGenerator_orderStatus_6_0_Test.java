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

public class EWrapperMsgGenerator_orderStatus_6_0_Test {

    @Test
    public void testOrderStatus() {
        // Test data
        int orderId = 12345;
        String status = "Filled";
        int filled = 100;
        int remaining = 0;
        double avgFillPrice = 150.75;
        int permId = 67890;
        int parentId = 0;
        double lastFillPrice = 150.75;
        int clientId = 54321;
        String whyHeld = "None";
        // Expected result
        String expected = "order status: orderId=" + orderId + " clientId=" + clientId + " permId=" + permId + " status=" + status + " filled=" + filled + " remaining=" + remaining + " avgFillPrice=" + avgFillPrice + " lastFillPrice=" + lastFillPrice + " parent Id=" + parentId + " whyHeld=" + whyHeld;
        // Actual result from the method under test
        String actual = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        // Assertion
        assertEquals(expected, actual);
    }

    @Test
    public void testOrderStatusWithPendingOrder() {
        // Test data for a pending order
        int orderId = 54321;
        String status = "Pending";
        int filled = 50;
        int remaining = 50;
        double avgFillPrice = 100.00;
        int permId = 11111;
        int parentId = 22222;
        double lastFillPrice = 99.50;
        int clientId = 33333;
        String whyHeld = "Regulatory Hold";
        // Expected result
        String expected = "order status: orderId=" + orderId + " clientId=" + clientId + " permId=" + permId + " status=" + status + " filled=" + filled + " remaining=" + remaining + " avgFillPrice=" + avgFillPrice + " lastFillPrice=" + lastFillPrice + " parent Id=" + parentId + " whyHeld=" + whyHeld;
        // Actual result from the method under test
        String actual = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        // Assertion
        assertEquals(expected, actual);
    }
}
