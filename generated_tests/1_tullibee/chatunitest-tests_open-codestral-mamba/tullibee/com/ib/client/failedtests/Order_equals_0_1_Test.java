package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Order_equals_0_1_Test {

    @Test
    void testEquals_sameObject() {
        Order order = new Order();
        assertEquals(order, order);
    }

    @Test
    void testEquals_null() {
        Order order = new Order();
        assertNotEquals(order, null);
    }

    @Test
    void testEquals_differentClass() {
        Order order = new Order();
        assertNotEquals(order, new Object());
    }

    @Test
    void testEquals_sameAttributes() {
        Order order1 = new Order();
        Order order2 = new Order();
        assertEquals(order1, order2);
    }

    @Test
    void testEquals_differentAttributes() {
        Order order1 = new Order();
        Order order2 = new Order();
        order2.m_permId = 1;
        assertNotEquals(order1, order2);
    }
}
