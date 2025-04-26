package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Order_equals_0_0_Test {

    @Test
    public void testEquals_SameReference() {
        Order order = new Order();
        assertTrue(order.equals(order), "Same reference should be equal");
    }

    @Test
    public void testEquals_NullObject() {
        Order order = new Order();
        assertFalse(order.equals(null), "Should not be equal to null");
    }

    @Test
    public void testEquals_DifferentClass() {
        Order order = new Order();
        String notAnOrder = "Not an Order";
        assertFalse(order.equals(notAnOrder), "Should not be equal to an object of a different class");
    }

    @Test
    public void testEquals_DifferentOrders_DifferentFields() {
        Order order1 = new Order();
        order1.m_permId = 1;
        order1.m_orderId = 1;
        Order order2 = new Order();
        // Different permId
        order2.m_permId = 2;
        assertFalse(order1.equals(order2), "Orders with different permId should not be equal");
    }

    @Test
    public void testEquals_SameOrders_SameFields() {
        Order order1 = new Order();
        order1.m_permId = 1;
        order1.m_orderId = 1;
        order1.m_clientId = 1;
        order1.m_totalQuantity = 100;
        order1.m_lmtPrice = 50.0;
        Order order2 = new Order();
        // Same permId
        order2.m_permId = 1;
        // Same orderId
        order2.m_orderId = 1;
        // Same clientId
        order2.m_clientId = 1;
        // Same totalQuantity
        order2.m_totalQuantity = 100;
        // Same lmtPrice
        order2.m_lmtPrice = 50.0;
        assertTrue(order1.equals(order2), "Orders with the same fields should be equal");
    }

    @Test
    public void testEquals_DifferentOrders_VaryingFields() {
        Order order1 = new Order();
        order1.m_permId = 1;
        order1.m_orderId = 1;
        order1.m_totalQuantity = 100;
        order1.m_lmtPrice = 50.0;
        order1.m_action = "BUY";
        Order order2 = new Order();
        // Same permId
        order2.m_permId = 1;
        // Same orderId
        order2.m_orderId = 1;
        // Same totalQuantity
        order2.m_totalQuantity = 100;
        // Same lmtPrice
        order2.m_lmtPrice = 50.0;
        // Different action
        order2.m_action = "SELL";
        assertFalse(order1.equals(order2), "Orders with different action should not be equal");
    }

    @Test
    public void testEquals_SameOrders_WithVectorParams() {
        Order order1 = new Order();
        order1.m_permId = 1;
        order1.m_orderId = 1;
        order1.m_algoParams = new Vector();
        order1.m_algoParams.add("param1");
        Order order2 = new Order();
        // Same permId
        order2.m_permId = 1;
        // Same orderId
        order2.m_orderId = 1;
        order2.m_algoParams = new Vector();
        // Same algoParams
        order2.m_algoParams.add("param1");
        assertTrue(order1.equals(order2), "Orders with the same algoParams should be equal");
    }

    @Test
    public void testEquals_DifferentOrders_WithVectorParams() {
        Order order1 = new Order();
        order1.m_permId = 1;
        order1.m_orderId = 1;
        order1.m_algoParams = new Vector();
        order1.m_algoParams.add("param1");
        Order order2 = new Order();
        // Same permId
        order2.m_permId = 1;
        // Same orderId
        order2.m_orderId = 1;
        order2.m_algoParams = new Vector();
        // Different algoParams
        order2.m_algoParams.add("param2");
        assertFalse(order1.equals(order2), "Orders with different algoParams should not be equal");
    }
}
