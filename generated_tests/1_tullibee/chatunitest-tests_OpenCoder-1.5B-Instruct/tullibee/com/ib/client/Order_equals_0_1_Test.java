// Test method
package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Order_equals_0_1_Test {

    @Test
    public void testEquals() {
        Order order = new Order();
        order.m_orderId = 12345;
        Order otherOrder = new Order();
        otherOrder.m_orderId = 12345;
        Assertions.assertTrue(order.equals(otherOrder));
    }

    @Test
    public void testNotEquals() {
        Order order = new Order();
        order.m_orderId = 12345;
        Order otherOrder = new Order();
        otherOrder.m_orderId = 99999;
        Assertions.assertFalse(order.equals(otherOrder));
    }
}
