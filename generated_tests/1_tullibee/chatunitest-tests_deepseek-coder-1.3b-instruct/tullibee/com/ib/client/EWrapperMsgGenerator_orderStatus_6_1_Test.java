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

public class EWrapperMsgGenerator_orderStatus_6_1_Test {

    @Test
    void testOrderStatus() {
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        int orderId = 123;
        String status = "FILLED";
        int filled = 2;
        int remaining = 3;
        double avgFillPrice = 4.5;
        int permId = 5;
        int parentId = 6;
        double lastFillPrice = 7.5;
        int clientId = 8;
        String whyHeld = "Test";
        String expected = "order status: orderId=123 clientId=8 permId=5 status=FILLED filled=2 remaining=3 avgFillPrice=4.5 lastFillPrice=7.5 parent Id=6 whyHeld=Test";
        assertEquals(expected, eWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld));
    }
}
