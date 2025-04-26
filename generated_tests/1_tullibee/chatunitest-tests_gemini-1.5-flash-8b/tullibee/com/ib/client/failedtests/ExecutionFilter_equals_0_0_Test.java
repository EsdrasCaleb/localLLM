package com.ib.client;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ExecutionFilter_equals_0_0_Test {

    @Test
    void testEquals_nullObject() {
        ExecutionFilter filter = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        assertFalse(filter.equals(null));
    }

    @Test
    void testEquals_differentObject() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        Object otherObject = new Object();
        assertFalse(filter1.equals(otherObject));
    }

    @Test
    void testEquals_sameObject() {
        ExecutionFilter filter = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        assertTrue(filter.equals(filter));
    }

    @Test
    void testEquals_differentClientId() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        ExecutionFilter filter2 = new ExecutionFilter(2, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_differentAcctCode() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        ExecutionFilter filter2 = new ExecutionFilter(1, "acct2", "time1", "symbol1", "secType1", "exchange1", "buy");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_differentTime() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        ExecutionFilter filter2 = new ExecutionFilter(1, "acct1", "time2", "symbol1", "secType1", "exchange1", "buy");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_differentSymbol() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        ExecutionFilter filter2 = new ExecutionFilter(1, "acct1", "time1", "symbol2", "secType1", "exchange1", "buy");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_differentFields_allMatch() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        ExecutionFilter filter2 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        assertTrue(filter1.equals(filter2));
    }

    @Test
    void testEquals_differentCase() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "buy");
        // Different case
        ExecutionFilter filter2 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "BUY");
        assertTrue(filter1.equals(filter2));
    }
}
