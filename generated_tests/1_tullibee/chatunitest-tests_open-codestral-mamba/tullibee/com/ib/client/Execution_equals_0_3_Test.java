package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Execution_equals_0_3_Test {

    @Test
    void testEquals() {
        Execution execution1 = new Execution();
        Execution execution2 = new Execution();
        execution1.m_execId = "123";
        execution2.m_execId = "123";
        assertEquals(true, execution1.equals(execution2));
        execution2.m_execId = "456";
        assertEquals(false, execution1.equals(execution2));
        execution2 = null;
        assertEquals(false, execution1.equals(execution2));
        execution2 = execution1;
        assertEquals(true, execution1.equals(execution2));
    }
}
