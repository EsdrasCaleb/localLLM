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

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_orderStatus_6_0_Test {

    // Test class
    @Test
    public void test_orderStatus_1() {
        // Arrange
        int orderId = 1;
        String status = "PARTIALLY_FILLED";
        int filled = 10;
        int remaining = 20;
        double avgFillPrice = 10.0;
        int permId = 10;
        int parentId = 10;
        double lastFillPrice = 10.0;
        int clientId = 10;
        String whyHeld = "HELD";
        // Act
        String actual = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        // Assert
        String expected = "order status: orderId=1 clientId=10 permId=10 status=PARTIALLY_FILLED filled=10 remaining=20 avgFillPrice=10.0 lastFillPrice=10.0 parent Id=10 whyHeld=HELD";
        assertEquals(expected, actual);
    }
}
