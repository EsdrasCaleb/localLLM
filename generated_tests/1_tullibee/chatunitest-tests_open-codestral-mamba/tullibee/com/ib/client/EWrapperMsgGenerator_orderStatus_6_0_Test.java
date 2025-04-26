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
        String status = "Filled";
        int filled = 100;
        int remaining = 0;
        double avgFillPrice = 150.50;
        int permId = 456;
        int parentId = 789;
        double lastFillPrice = 150.50;
        int clientId = 987;
        String whyHeld = "None";
        String expectedStatusMessage = "order status: orderId=123 clientId=987 permId=456 status=Filled filled=100 remaining=0 avgFillPrice=150.5 lastFillPrice=150.5 parent Id=789 whyHeld=None";
        String actualStatusMessage = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        assertEquals(expectedStatusMessage, actualStatusMessage);
    }
}
