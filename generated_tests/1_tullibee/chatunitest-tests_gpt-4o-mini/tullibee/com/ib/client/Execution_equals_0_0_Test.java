package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_0_Test {

    private Execution execution1;

    private Execution execution2;

    private Execution execution3;

    @BeforeEach
    public void setUp() {
        execution1 = new Execution(1, 100, "exec123", "2023-01-01", "acct1", "NYSE", "BUY", 10, 100.0, 1, 0, 0, 0.0);
        execution2 = new Execution(2, 200, "exec123", "2023-01-02", "acct2", "NASDAQ", "SELL", 15, 150.0, 2, 0, 0, 0.0);
        execution3 = new Execution(3, 300, "exec456", "2023-01-03", "acct3", "AMEX", "BUY", 20, 200.0, 3, 0, 0, 0.0);
    }

    @Test
    public void testEquals_NullObject() {
        assertFalse(execution1.equals(null));
    }

    @Test
    public void testEquals_SameInstance() {
        assertTrue(execution1.equals(execution1));
    }

    @Test
    public void testEquals_SameExecId() {
        assertTrue(execution1.equals(execution2));
    }

    @Test
    public void testEquals_DifferentExecId() {
        assertFalse(execution1.equals(execution3));
    }
}
