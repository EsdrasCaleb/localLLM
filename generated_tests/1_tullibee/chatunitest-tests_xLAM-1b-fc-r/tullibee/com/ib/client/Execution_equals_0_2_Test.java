package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Execution_equals_0_2_Test {

    @Test
    void testEquals() {
        Execution execution1 = new Execution();
        execution1.m_execId = "test1";
        Execution execution2 = new Execution();
        execution2.m_execId = "test2";
        assertFalse(execution1.equals(execution2));
        execution2.m_execId = "test1";
        assertTrue(execution1.equals(execution2));
    }
}
