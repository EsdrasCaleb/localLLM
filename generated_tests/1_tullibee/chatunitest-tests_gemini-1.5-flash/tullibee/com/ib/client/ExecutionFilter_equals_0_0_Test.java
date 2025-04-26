package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_0_Test {

    @Test
    void testEquals_Null() {
        ExecutionFilter filter = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertFalse(filter.equals(null));
    }

    @Test
    void testEquals_SameObject() {
        ExecutionFilter filter = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertTrue(filter.equals(filter));
    }

    @Test
    void testEquals_EqualObjects() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertTrue(filter1.equals(filter2));
    }

    @Test
    void testEquals_DifferentClientId() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(2, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_DifferentAcctCode() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ACCT2", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_DifferentAcctCodeCaseInsensitive() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "acct1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertTrue(filter1.equals(filter2));
    }

    @Test
    void testEquals_DifferentTime() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ACCT1", "11:00", "AAPL", "STK", "NYSE", "BUY");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_DifferentTimeCaseInsensitive() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "buy");
        assertTrue(filter1.equals(filter2));
    }

    @Test
    void testEquals_DifferentSymbol() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ACCT1", "10:00", "MSFT", "STK", "NYSE", "BUY");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    void testEquals_DifferentSecType() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ACCT1", "10:00", "AAPL", "OPT", "NYSE", "BUY");
        assertFalse(filter1.equals(filter2));
    }
}
