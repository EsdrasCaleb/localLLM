package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Execution_equals_0_1_Test {

    @Test
    void testEquals_Null() {
        Execution execution = new Execution();
        assertFalse(execution.equals(null));
    }

    @Test
    void testEquals_Self() {
        Execution execution = new Execution();
        assertTrue(execution.equals(execution));
    }

    @Test
    void testEquals_SameExecId() {
        Execution execution1 = new Execution();
        try {
            Field execIdField = Execution.class.getDeclaredField("m_execId");
            execIdField.setAccessible(true);
            execIdField.set(execution1, "sameId");
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set m_execId field: " + e.getMessage());
        }
        Execution execution2 = new Execution();
        try {
            Field execIdField = Execution.class.getDeclaredField("m_execId");
            execIdField.setAccessible(true);
            execIdField.set(execution2, "sameId");
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set m_execId field: " + e.getMessage());
        }
        assertTrue(execution1.equals(execution2));
    }

    @Test
    void testEquals_DifferentExecId() {
        Execution execution1 = new Execution();
        try {
            Field execIdField = Execution.class.getDeclaredField("m_execId");
            execIdField.setAccessible(true);
            execIdField.set(execution1, "id1");
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set m_execId field: " + e.getMessage());
        }
        Execution execution2 = new Execution();
        try {
            Field execIdField = Execution.class.getDeclaredField("m_execId");
            execIdField.setAccessible(true);
            execIdField.set(execution2, "id2");
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set m_execId field: " + e.getMessage());
        }
        assertFalse(execution1.equals(execution2));
    }
}
