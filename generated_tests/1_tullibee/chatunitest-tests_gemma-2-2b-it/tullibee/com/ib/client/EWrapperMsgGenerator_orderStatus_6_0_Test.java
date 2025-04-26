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
    void testOrderStatus() {
        int orderId = 1;
        String status = "Filled";
        int filled = 10;
        int remaining = 20;
        double avgFillPrice = 10.0;
        int permId = 1234;
        int parentId = 5678;
        double lastFillPrice = 15.0;
        int clientId = 9876;
        String whyHeld = "Pending order";
        String actualOrderStatus = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        assertEquals("order status: orderId=1 clientId=9876 permId=1234 status=Filled filled=10 remaining=20 avgFillPrice=10.0 lastFillPrice=15.0 parent Id=5678 whyHeld=Pending order", actualOrderStatus);
    }
}
