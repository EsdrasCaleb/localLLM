package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class // ... more tests covering various combinations of field values
Order_equals_0_0_Test {

    @Test
    void testEquals_sameObject() {
        Order order = new Order();
        assertTrue(order.equals(order));
    }

    @Test
    void testEquals_nullObject() {
        Order order = new Order();
        assertFalse(order.equals(null));
    }

    @Test
    void testEquals_allFieldsMatch() {
        Order order1 = new Order();
        order1.m_orderId = 1;
        order1.m_clientId = 2;
        order1.m_totalQuantity = 10;
        order1.m_lmtPrice = 100.0;
        // ... set other fields for comparison
        // crucial for a meaningful test
        order1.m_permId = 123;
        Order order2 = new Order();
        order2.m_orderId = 1;
        order2.m_clientId = 2;
        order2.m_totalQuantity = 10;
        order2.m_lmtPrice = 100.0;
        // ... set other fields to match order1
        order2.m_permId = 123;
        assertTrue(order1.equals(order2));
    }

    @Test
    void testEquals_stringFieldsMatch() {
        Order order1 = new Order();
        order1.m_action = "buy";
        Order order2 = new Order();
        order2.m_action = "buy";
        assertTrue(order1.equals(order2));
    }

    @Test
    void testEquals_vectorFieldsMatch() {
        Order order1 = new Order();
        Vector<String> params1 = new Vector<>();
        params1.add("param1");
        order1.m_algoParams = params1;
        Order order2 = new Order();
        Vector<String> params2 = new Vector<>();
        params2.add("param1");
        order2.m_algoParams = params2;
        assertTrue(order1.equals(order2));
    }
}
