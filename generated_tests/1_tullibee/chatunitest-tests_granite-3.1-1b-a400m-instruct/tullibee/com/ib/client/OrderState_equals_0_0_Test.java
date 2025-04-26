package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_0_Test {

    @Test
    public void testEquals() {
        OrderState state1 = new OrderState("Open", "100", "50", "1000", 0.01, 0.005, 0.1, "USD", "Warning");
        OrderState state2 = new OrderState("Open", "100", "50", "1000", 0.01, 0.005, 0.1, "USD", "Warning");
        OrderState state3 = new OrderState("Close", "50", "25", "500", 0.005, 0.002, 0.05, "EUR", "Warning");
        assertTrue(state1.equals(state2));
        assertFalse(state1.equals(state3));
    }
}
