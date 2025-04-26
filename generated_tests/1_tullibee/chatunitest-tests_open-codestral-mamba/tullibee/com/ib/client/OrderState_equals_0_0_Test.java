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
        // Create two instances of OrderState with the same values
        OrderState state1 = new OrderState("state1", "initMargin1", "maintMargin1", "equityWithLoan1", 10.0, 5.0, 20.0, "USD", "warningText1");
        OrderState state2 = new OrderState("state1", "initMargin1", "maintMargin1", "equityWithLoan1", 10.0, 5.0, 20.0, "USD", "warningText1");
        // Assert that the two instances are equal
        assertTrue(state1.equals(state2));
        // Modify one field of state2
        state2.m_status = "state2";
        // Assert that the two instances are not equal
        assertFalse(state1.equals(state2));
    }
}
