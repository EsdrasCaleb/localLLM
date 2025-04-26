package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_2_Test {

    @Test
    public void testEquals_Self() {
        Execution e1 = new Execution(1, 1, "exec1", "time1", "acct1", "exchange1", "side1", 10, 100.0, 1, 0, 0, 0);
        assertTrue(e1.equals(e1));
    }

    @Test
    public void testEquals_DifferentObjects() {
        Execution e1 = new Execution(1, 1, "exec1", "time1", "acct1", "exchange1", "side1", 10, 100.0, 1, 0, 0, 0);
        Execution e2 = new Execution(1, 1, "exec2", "time1", "acct1", "exchange1", "side1", 10, 100.0, 1, 0, 0, 0);
        assertFalse(e1.equals(e2));
    }

    @Test
    public void testEquals_MatchingExecId() {
        Execution e1 = new Execution(1, 1, "exec1", "time1", "acct1", "exchange1", "side1", 10, 100.0, 1, 0, 0, 0);
        Execution e2 = new Execution(1, 1, "exec1", "time1", "acct1", "exchange1", "side1", 10, 100.0, 1, 0, 0, 0);
        assertTrue(e1.equals(e2));
    }
}
