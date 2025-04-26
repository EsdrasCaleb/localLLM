package com.ib.client;

import java.lang.reflect.Field;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Order_equals_0_0_Test {

    // Dummy Util class for testing purposes.  Replace with your actual Util class.
    static class Util {

        public static int StringCompare(String s1, String s2) {
            return s1 == null ? (s2 == null ? 0 : -1) : (s2 == null ? 1 : s1.compareTo(s2));
        }

        public static boolean VectorEqualsUnordered(Vector v1, Vector v2) {
            if (v1 == null || v2 == null)
                return v1 == v2;
            // Replace with your actual unordered comparison if needed.
            return v1.equals(v2);
        }
    }

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
    void testEquals_differentObjects_permIdMatch() throws Exception {
        Order order1 = new Order();
        Order order2 = new Order();
        // Set permId to be the same for both objects
        setField(order1, "m_permId", 123);
        setField(order2, "m_permId", 123);
        assertTrue(order1.equals(order2));
    }

    @Test
    void testEquals_differentObjects_permIdMismatch_allOtherFieldsMatch() throws Exception {
        Order order1 = createOrderWithSampleData(123);
        Order order2 = createOrderWithSampleData(456);
        assertFalse(order1.equals(order2));
    }

    @Test
    void testEquals_differentObjects_allFieldsMatch() throws Exception {
        Order order1 = createOrderWithSampleData(123);
        Order order2 = createOrderWithSampleData(123);
        assertTrue(order1.equals(order2));
    }

    @Test
    void testEquals_differentObjects_stringFieldsMismatch() throws Exception {
        Order order1 = createOrderWithSampleData(123);
        Order order2 = createOrderWithSampleData(123);
        // Change a string field
        setField(order2, "m_action", "SELL");
        assertFalse(order1.equals(order2));
    }

    @Test
    void testEquals_differentObjects_vectorFieldsMismatch() throws Exception {
        Order order1 = createOrderWithSampleData(123);
        Order order2 = createOrderWithSampleData(123);
        Vector<String> v1 = new Vector<>();
        v1.add("param1");
        Vector<String> v2 = new Vector<>();
        v2.add("param2");
        setField(order1, "m_algoParams", v1);
        setField(order2, "m_algoParams", v2);
        assertFalse(order1.equals(order2));
    }

    private Order createOrderWithSampleData(int permId) throws Exception {
        Order order = new Order();
        setField(order, "m_permId", permId);
        setField(order, "m_orderId", 1);
        setField(order, "m_clientId", 2);
        setField(order, "m_totalQuantity", 100);
        setField(order, "m_lmtPrice", 10.0);
        setField(order, "m_auxPrice", 11.0);
        setField(order, "m_action", "BUY");
        setField(order, "m_orderType", "LMT");
        // ... set other fields as needed ...
        return order;
    }

    private void setField(Object obj, String fieldName, Object value) throws Exception {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }
}
