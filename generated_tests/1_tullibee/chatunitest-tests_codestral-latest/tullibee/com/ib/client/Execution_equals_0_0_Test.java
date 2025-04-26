package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_0_Test {

    @InjectMocks
    private Execution execution;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        execution = new Execution(1, 101, "exec123", "2023-10-01", "acct123", "NYSE", "BUY", 100, 50.0, 1001, 0, 50, 50.0);
    }

    @Test
    public void testEquals_SameObject() {
        assertTrue(execution.equals(execution));
    }

    @Test
    public void testEquals_NullObject() {
        assertFalse(execution.equals(null));
    }

    @Test
    public void testEquals_DifferentExecId() {
        Execution otherExecution = new Execution(2, 102, "exec456", "2023-10-02", "acct456", "NASDAQ", "SELL", 200, 60.0, 1002, 0, 60, 60.0);
        assertFalse(execution.equals(otherExecution));
    }

    @Test
    public void testEquals_SameExecId() {
        Execution otherExecution = new Execution(1, 101, "exec123", "2023-10-01", "acct123", "NYSE", "BUY", 100, 50.0, 1001, 0, 50, 50.0);
        assertTrue(execution.equals(otherExecution));
    }
}
