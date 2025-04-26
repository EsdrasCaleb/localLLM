package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Order_equals_0_3_Test {

    @Test
    public void testEquals_ObjectNull() {
        Order order = new Order();
        Object obj = null;
        assertFalse(order.equals(obj));
    }

    @Test
    public void testEquals_ObjectNotOrder() {
        Order order = new Order();
        Object obj = new Object();
        assertFalse(order.equals(obj));
    }

    @Test
    public void testEquals_ObjectOfDifferentClass() {
        Order order = new Order();
        Object obj = new String();
        assertFalse(order.equals(obj));
    }

    @Test
    public void testEquals_ObjectOfSameClass() {
        Order order1 = new Order();
        Order order2 = new Order();
        assertTrue(order1.equals(order2));
    }
}
