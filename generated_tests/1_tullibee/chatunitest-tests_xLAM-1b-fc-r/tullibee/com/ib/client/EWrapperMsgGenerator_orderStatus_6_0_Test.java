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
        int orderId = 123;
        String status = "open";
        int filled = 20;
        int remaining = 5;
        double avgFillPrice = 10.50;
        int permId = 456;
        int parentId = 789;
        double lastFillPrice = 8.50;
        int clientId = 1234;
        String whyHeld = "reason";
        String expected = "order status: orderId=" + orderId + " clientId=" + clientId + " permId=" + permId + " status=" + status + " filled=" + filled + " remaining=" + remaining + " avgFillPrice=" + avgFillPrice + " lastFillPrice=" + lastFillPrice + " parent Id=" + parentId + " whyHeld=" + whyHeld;
        String result = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        assertEquals(expected, result);
    }
}
