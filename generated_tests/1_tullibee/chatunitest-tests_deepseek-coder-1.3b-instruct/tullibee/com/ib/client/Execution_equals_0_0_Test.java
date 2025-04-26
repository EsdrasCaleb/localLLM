package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_0_Test {

    @Test
    public void testEquals() {
        Execution execution1 = new Execution();
        Execution execution2 = new Execution();
        execution1.m_execId = "testId";
        execution2.m_execId = "testId";
        assertEquals(execution1, execution2);
        assertTrue(execution1.equals(execution2));
    }

    @Test
    public void testNotEquals() {
        Execution execution1 = new Execution();
        Execution execution2 = new Execution();
        execution1.m_execId = "testId";
        execution2.m_execId = "otherId";
        assertFalse(execution1.equals(execution2));
        assertFalse(execution1.equals(null));
    }
}
