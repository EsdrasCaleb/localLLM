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
        int orderId = 1;
        String status = "Filled";
        int filled = 100;
        int remaining = 0;
        double avgFillPrice = 50.5;
        int permId = 12345;
        int parentId = 67890;
        double lastFillPrice = 50.5;
        int clientId = 101;
        String whyHeld = "None";
        String expected = "order status: orderId=1 clientId=101 permId=12345 status=Filled filled=100 remaining=0 avgFillPrice=50.5 lastFillPrice=50.5 parent Id=67890 whyHeld=None";
        String result = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        assertEquals(expected, result);
    }
}
