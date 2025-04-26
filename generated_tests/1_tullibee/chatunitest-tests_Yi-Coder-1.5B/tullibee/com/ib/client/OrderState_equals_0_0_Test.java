package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_0_Test {

    @Test
    void testEquals() {
        OrderState state1 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        OrderState state2 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        OrderState state3 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        OrderState state4 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        OrderState state5 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        OrderState state6 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        OrderState state7 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        OrderState state8 = new OrderState("ACTIVE", "0.02", "0.02", "0.02", 0.02, 0.02, 0.02, "USD", "");
        assertFalse(state1.equals(state2));
        assertFalse(state2.equals(state3));
        assertFalse(state3.equals(state4));
        assertTrue(state4.equals(state5));
        assertFalse(state5.equals(state6));
        assertTrue(state6.equals(state7));
        assertFalse(state7.equals(state8));
    }
}
