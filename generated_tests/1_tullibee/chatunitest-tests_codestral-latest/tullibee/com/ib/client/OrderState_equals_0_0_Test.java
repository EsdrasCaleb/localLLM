package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_0_Test {

    private OrderState orderState1;

    private OrderState orderState2;

    @BeforeEach
    void setUp() {
        orderState1 = new OrderState("active", "1000", "500", "1500", 1.5, 0.5, 2.5, "USD", "Warning");
        orderState2 = new OrderState("active", "1000", "500", "1500", 1.5, 0.5, 2.5, "USD", "Warning");
    }

    @Test
    void testEquals_SameObject() {
        assertTrue(orderState1.equals(orderState1));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(orderState1.equals(null));
    }

    @Test
    void testEquals_DifferentCommission() {
        orderState2.m_commission = 2.0;
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_DifferentMinCommission() {
        orderState2.m_minCommission = 1.0;
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_DifferentMaxCommission() {
        orderState2.m_maxCommission = 3.0;
        assertFalse(orderState1.equals(orderState2));
    }

    @Test
    void testEquals_AllFieldsEqual() {
        assertTrue(orderState1.equals(orderState2));
    }
}
