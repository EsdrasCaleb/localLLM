package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_0_Test {

    private Execution execution;

    @BeforeEach
    public void setUp() {
        execution = new Execution();
        execution.m_execId = "exec123";
    }

    @Test
    public void testEquals_withNullObject() {
        assertFalse(execution.equals(null));
    }

    @Test
    public void testEquals_withSameInstance() {
        assertTrue(execution.equals(execution));
    }

    @Test
    public void testEquals_withDifferentInstanceSameExecId() {
        Execution otherExecution = new Execution();
        otherExecution.m_execId = "exec123";
        assertTrue(execution.equals(otherExecution));
    }

    @Test
    public void testEquals_withDifferentInstanceDifferentExecId() {
        Execution otherExecution = new Execution();
        otherExecution.m_execId = "exec456";
        assertFalse(execution.equals(otherExecution));
    }
}
