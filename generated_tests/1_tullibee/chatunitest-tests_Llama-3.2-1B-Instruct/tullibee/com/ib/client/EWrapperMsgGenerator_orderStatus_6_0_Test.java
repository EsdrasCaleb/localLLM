package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_orderStatus_6_0_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @InjectMocks
    private EWrapperMsgGenerator instance;

    @Test
    public void testOrderStatus() {
        // Arrange
        int orderId = 1;
        String status = "Filled";
        int filled = 10;
        int remaining = 20;
        double avgFillPrice = 50.0;
        int permId = 1;
        int parentId = 1;
        double lastFillPrice = 40.0;
        int clientId = 1;
        String whyHeld = "Held";
        // Act
        String result = instance.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        // Assert
        assertEquals("order status: orderId=" + orderId + " clientId=" + clientId + " permId=" + permId + " status=" + status + " filled=" + filled + " remaining=" + remaining + " avgFillPrice=" + avgFillPrice + " lastFillPrice=" + lastFillPrice + " parent Id=" + parentId + " whyHeld=" + whyHeld, result);
    }
}
