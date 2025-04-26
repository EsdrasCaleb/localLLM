package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_0_Test {

    // Dummy Util class for testing purposes. Replace with your actual Util class.
    static class Util {

        public static int StringCompare(String s1, String s2) {
            if (s1 == null && s2 == null)
                return 0;
            if (s1 == null)
                return -1;
            if (s2 == null)
                return 1;
            return s1.compareTo(s2);
        }
    }

    @Test
    void testEquals_sameObject() {
        OrderState state1 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        assertTrue(state1.equals(state1));
    }

    @Test
    void testEquals_nullObject() {
        OrderState state1 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        assertFalse(state1.equals(null));
    }

    @Test
    void testEquals_differentObject() {
        OrderState state1 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        OrderState state2 = new OrderState("closed", "200", "100", "300", 2.0, 1.0, 4.0, "EUR", "Another warning!");
        assertFalse(state1.equals(state2));
    }

    @Test
    void testEquals_differentCommission() {
        OrderState state1 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        OrderState state2 = new OrderState("open", "100", "50", "150", 1.1, 0.5, 2.0, "USD", "Warning!");
        assertFalse(state1.equals(state2));
    }

    @Test
    void testEquals_differentMinCommission() {
        OrderState state1 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        OrderState state2 = new OrderState("open", "100", "50", "150", 1.0, 0.6, 2.0, "USD", "Warning!");
        assertFalse(state1.equals(state2));
    }

    @Test
    void testEquals_differentMaxCommission() {
        OrderState state1 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        OrderState state2 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.1, "USD", "Warning!");
        assertFalse(state1.equals(state2));
    }

    @Test
    void testEquals_differentStringFields() {
        OrderState state1 = new OrderState("open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        // Different case
        OrderState state2 = new OrderState("Open", "100", "50", "150", 1.0, 0.5, 2.0, "USD", "Warning!");
        assertFalse(state1.equals(state2));
    }
}
