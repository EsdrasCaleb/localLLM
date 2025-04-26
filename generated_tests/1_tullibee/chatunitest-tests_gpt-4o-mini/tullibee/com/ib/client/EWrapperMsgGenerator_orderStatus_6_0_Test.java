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
        // Test case 1: Normal values
        String result1 = EWrapperMsgGenerator.orderStatus(1, "Filled", 100, 0, 50.0, 123, 0, 50.0, 456, "None");
        assertEquals("order status: orderId=1 clientId=456 permId=123 status=Filled filled=100 remaining=0 avgFillPrice=50.0 lastFillPrice=50.0 parent Id=0 whyHeld=None", result1);
        // Test case 2: No shares filled
        String result2 = EWrapperMsgGenerator.orderStatus(2, "Pending", 0, 100, 0.0, 124, 0, 0.0, 457, "Waiting");
        assertEquals("order status: orderId=2 clientId=457 permId=124 status=Pending filled=0 remaining=100 avgFillPrice=0.0 lastFillPrice=0.0 parent Id=0 whyHeld=Waiting", result2);
        // Test case 3: All parameters at their limits
        String result3 = EWrapperMsgGenerator.orderStatus(Integer.MAX_VALUE, "Completed", Integer.MAX_VALUE, Integer.MAX_VALUE, Double.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE, Double.MAX_VALUE, Integer.MAX_VALUE, "Maxed Out");
        assertEquals("order status: orderId=2147483647 clientId=2147483647 permId=2147483647 status=Completed filled=2147483647 remaining=2147483647 avgFillPrice=1.7976931348623157E308 lastFillPrice=1.7976931348623157E308 parent Id=2147483647 whyHeld=Maxed Out", result3);
        // Test case 4: Empty status and whyHeld
        String result4 = EWrapperMsgGenerator.orderStatus(3, "", 50, 50, 25.0, 125, 1, 25.0, 458, "");
        assertEquals("order status: orderId=3 clientId=458 permId=125 status= filled=50 remaining=50 avgFillPrice=25.0 lastFillPrice=25.0 parent Id=1 whyHeld=", result4);
    }
}
