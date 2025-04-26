package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_1_Test {

    @Test
    public void testEqualsNull() {
        Execution execution1 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        Execution execution2 = null;
        assertTrue(execution1.equals(execution2));
    }

    @Test
    public void testEqualsSameObject() {
        Execution execution1 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        Execution execution2 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        assertTrue(execution1.equals(execution2));
    }

    @Test
    public void testEqualsDifferentObject() {
        Execution execution1 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        Execution execution2 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        assertFalse(execution1.equals(execution2));
    }

    @Test
    public void testEqualsDifferentExecId() {
        Execution execution1 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        Execution execution2 = new Execution(1, 3, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        assertFalse(execution1.equals(execution2));
    }

    @Test
    public void testEqualsDifferentTime() {
        Execution execution1 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        Execution execution2 = new Execution(1, 2, "EXEC1", "2022-01-02", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        assertFalse(execution1.equals(execution2));
    }

    @Test
    public void testEqualsDifferentAcctNumber() {
        Execution execution1 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT1", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        Execution execution2 = new Execution(1, 2, "EXEC1", "2022-01-01", "ACCT2", "EXCH1", "BUY", 100, 10.0, 1, 1, 1, 1);
        assertFalse(execution1.equals(execution2));
    }
}
