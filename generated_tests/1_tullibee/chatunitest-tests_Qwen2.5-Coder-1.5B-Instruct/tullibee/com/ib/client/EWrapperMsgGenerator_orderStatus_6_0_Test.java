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

class EWrapperMsgGenerator_orderStatus_6_0_Test {

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    void setUp() {
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
    }

    @Test
    void testOrderStatus() {
        // Arrange
        int orderId = 12345;
        String status = "open";
        int filled = 10;
        int remaining = 90;
        double avgFillPrice = 10.5;
        int permId = 67890;
        int parentId = 54321;
        double lastFillPrice = 11.2;
        int clientId = 98765;
        String whyHeld = "no reason";
        // Act
        String result = eWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        // Assert
        assertEquals("order status: orderId=12345 clientId=98765 permId=67890 status=open filled=10 remaining=90 avgFillPrice=10.5 lastFillPrice=11.2 parent Id=54321 whyHeld=no reason", result);
    }
}
