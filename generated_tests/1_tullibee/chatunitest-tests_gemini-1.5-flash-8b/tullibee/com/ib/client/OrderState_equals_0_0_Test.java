package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_0_Test {

    @Test
    void testEquals_sameObject() {
        OrderState orderState = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        assertTrue(orderState.equals(orderState));
    }

    @Test
    void testEquals_null() {
        OrderState orderState = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        assertFalse(orderState.equals(null));
    }

    @Test
    void testEquals_differentCommission() {
        OrderState orderState1 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        OrderState orderState2 = new OrderState("open", "100", "200", "300", 11.0, 5.0, 15.0, "USD", "No Warning");
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_differentMinCommission() {
        OrderState orderState1 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        OrderState orderState2 = new OrderState("open", "100", "200", "300", 10.0, 6.0, 15.0, "USD", "No Warning");
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_differentMaxCommission() {
        OrderState orderState1 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        OrderState orderState2 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 16.0, "USD", "No Warning");
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_differentStatus() {
        OrderState orderState1 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        OrderState orderState2 = new OrderState("closed", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_differentFields() {
        OrderState orderState1 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        OrderState orderState2 = new OrderState("open", "200", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_allFieldsEqual() {
        OrderState orderState1 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        OrderState orderState2 = new OrderState("open", "100", "200", "300", 10.0, 5.0, 15.0, "USD", "No Warning");
        assertTrue(orderState1.equals(orderState2));
    }
}
