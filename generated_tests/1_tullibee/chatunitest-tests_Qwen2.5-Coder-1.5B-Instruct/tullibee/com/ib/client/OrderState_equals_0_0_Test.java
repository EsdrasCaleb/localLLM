package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_0_Test {

    @Test
    public void testEquals() throws Exception {
        // Create mock objects for the OrderState class
        OrderState orderState1 = new OrderState("active", "100", "200", "300", 0.1, 0.05, 0.15, "USD", "No warnings");
        OrderState orderState2 = new OrderState("active", "100", "200", "300", 0.1, 0.05, 0.15, "USD", "No warnings");
        OrderState orderState3 = new OrderState("inactive", "400", "300", "200", 0.2, 0.06, 0.25, "EUR", "Low risk");
        // Verify that equal states return true
        assertTrue(orderState1.equals(orderState2));
        // Verify that unequal states return false
        assertFalse(orderState1.equals(orderState3));
    }
}
