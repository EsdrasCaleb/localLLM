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
class EWrapperMsgGenerator_orderStatus_6_0_Test {

    @Mock
    private EWrapperMsgGenerator wrapperMsgGenerator;

    @InjectMocks
    private EWrapperMsgGenerator underTest;

    @Test
    void testOrderStatus_ValidParams() {
        // Arrange
        int orderId = 1;
        String status = "Filled";
        int filled = 100;
        int remaining = 200;
        double avgFillPrice = 10.0;
        int permId = 1;
        int parentId = 1;
        double lastFillPrice = 15.0;
        int clientId = 1;
        String whyHeld = "Held for short-term";
        // Act
        String result = underTest.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        // Assert
        assert result.equals("order status: orderId=1 clientId=1 permId=1 status=Filled filled=100 remaining=200 avgFillPrice=10.0 lastFillPrice=15.0 parent Id=1 whyHeld=Held for short-term");
    }
}
