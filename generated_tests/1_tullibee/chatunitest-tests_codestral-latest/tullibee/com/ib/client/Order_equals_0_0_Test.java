package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Order_equals_0_0_Test {

    private Order order1;

    private Order order2;

    @BeforeEach
    void setUp() {
        order1 = new Order();
        order2 = new Order();
    }

    @Test
    void testEquals_SameObject() {
        assertTrue(order1.equals(order1));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(order1.equals(null));
    }

    @Test
    void testEquals_SamePermId() {
        order1.m_permId = 1;
        order2.m_permId = 1;
        assertTrue(order1.equals(order2));
    }
}
