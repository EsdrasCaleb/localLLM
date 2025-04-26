package com.ib.client;

import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_2_Test {

    @Test
    public void testEquals() {
        // Create two OrderState instances
        OrderState state1 = new OrderState("active", "10.0", "20.0", "10.0", 100.0, 100.0, 100.0, "10.0", "10.0");
        OrderState state2 = new OrderState("active", "10.0", "20.0", "10.0", 100.0, 100.0, 100.0, "10.0", "10.0");
        // Create a third OrderState instance
        OrderState state3 = new OrderState("inactive", "30.0", "40.0", "10.0", 150.0, 150.0, 150.0, "10.0", "10.0");
        // Test equals() method
        assertTrue(state1.equals(state2));
        assertTrue(state1.equals(state3));
        // Test not equals()
        assertFalse(state1.equals(state3));
    }
}
