package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_0_Test {

    @Test
    void testEquals() {
        OrderState state1 = new OrderState("Active", "10.0", "20.0", "1000.0", 1.0, 0.5, 2.0, "USD", "Warning");
        OrderState state2 = new OrderState("Active", "10.0", "20.0", "1000.0", 1.0, 0.5, 2.0, "USD", "Warning");
        OrderState state3 = new OrderState("Pending", "10.0", "20.0", "1000.0", 1.0, 0.5, 2.0, "USD", "Warning");
        OrderState state4 = new OrderState("Active", "10.0", "20.0", "1000.0", 1.0, 0.5, 2.0, "USD", "Warning");
        OrderState state5 = new OrderState("Pending", "10.0", "20.0", "1000.0", 1.0, 0.5, 2.0, "USD", "Warning");
        assertEquals(state1, state2);
        assertEquals(state1, state3);
        assertEquals(state1, state4);
        assertEquals(state2, state3);
        assertEquals(state2, state4);
    }
}
