package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ExecutionFilter_equals_0_0_Test {

    @InjectMocks
    private ExecutionFilter executionFilter;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        executionFilter = new ExecutionFilter(1, "A123", "10:00", "AAPL", "STK", "NYSE", "BUY");
    }

    @Test
    void testEquals_SameObject() {
        assertTrue(executionFilter.equals(executionFilter));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(executionFilter.equals(null));
    }

    @Test
    void testEquals_EqualObjects() {
        ExecutionFilter other = new ExecutionFilter(1, "A123", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertTrue(executionFilter.equals(other));
    }

    @Test
    void testEquals_DifferentClientId() {
        ExecutionFilter other = new ExecutionFilter(2, "A123", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertFalse(executionFilter.equals(other));
    }

    @Test
    void testEquals_DifferentAcctCode() {
        ExecutionFilter other = new ExecutionFilter(1, "B456", "10:00", "AAPL", "STK", "NYSE", "BUY");
        assertFalse(executionFilter.equals(other));
    }

    @Test
    void testEquals_DifferentTime() {
        ExecutionFilter other = new ExecutionFilter(1, "A123", "11:00", "AAPL", "STK", "NYSE", "BUY");
        assertFalse(executionFilter.equals(other));
    }

    @Test
    void testEquals_DifferentSymbol() {
        ExecutionFilter other = new ExecutionFilter(1, "A123", "10:00", "GOOGL", "STK", "NYSE", "BUY");
        assertFalse(executionFilter.equals(other));
    }

    @Test
    void testEquals_DifferentSecType() {
        ExecutionFilter other = new ExecutionFilter(1, "A123", "10:00", "AAPL", "OPT", "NYSE", "BUY");
        assertFalse(executionFilter.equals(other));
    }

    @Test
    void testEquals_DifferentExchange() {
        ExecutionFilter other = new ExecutionFilter(1, "A123", "10:00", "AAPL", "STK", "NASDAQ", "BUY");
        assertFalse(executionFilter.equals(other));
    }

    @Test
    void testEquals_DifferentSide() {
        ExecutionFilter other = new ExecutionFilter(1, "A123", "10:00", "AAPL", "STK", "NYSE", "SELL");
        assertFalse(executionFilter.equals(other));
    }
}
