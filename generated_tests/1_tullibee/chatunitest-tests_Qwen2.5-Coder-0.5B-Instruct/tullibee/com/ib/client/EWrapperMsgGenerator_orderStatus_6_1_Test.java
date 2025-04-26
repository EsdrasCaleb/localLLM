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

class EWrapperMsgGenerator_orderStatus_6_1_Test {

    @Test
    void testOrderStatus() {
        // Arrange
        int orderId = 123;
        String status = "Shipped";
        int filled = 50;
        int remaining = 40;
        double avgFillPrice = 10.99;
        int permId = 456;
        int parentId = 789;
        double lastFillPrice = 15.77;
        int clientId = 901;
        String whyHeld = "Delivered";
        // Act
        String result = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        // Assert
        assertEquals("order status: orderId=123 clientId=901 permId=789 status=Shipped filled=50 remaining=40 avgFillPrice=10.99 lastFillPrice=15.77", result);
    }
}
