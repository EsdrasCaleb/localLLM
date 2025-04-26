package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_1_Test {

    @Test
    void testEqualsNull() {
        ExecutionFilter filter1 = new ExecutionFilter();
        ExecutionFilter filter2 = null;
        assertEquals(false, filter1.equals(filter2));
    }

    @Test
    void testEqualsSameObject() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ABC", "2023-03-01", "AAPL", "STOCK", "NYSE", "BUY");
        ExecutionFilter filter2 = filter1;
        assertEquals(true, filter1.equals(filter2));
    }

    @Test
    void testEqualsDifferentObjects() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ABC", "2023-03-01", "AAPL", "STOCK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ABC", "2023-03-02", "AAPL", "STOCK", "NYSE", "SELL");
        assertEquals(false, filter1.equals(filter2));
    }
}
