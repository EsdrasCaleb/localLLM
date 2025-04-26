package com.ib.client;

import java.util.Arrays;
import java.util.Collection;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Execution_equals_0_0_Test {

    private Execution m_theExecution;

    @BeforeEach
    void setup() {
        m_theExecution = new Execution(123, 456, "0123456789", "11:11:11", "ACCT123456789", "BATS", "B", 100, 1.0, 1111, 1, 100, 1.0);
    }

    @Test
    void testEquals_Null() {
        Assertions.assertFalse(m_theExecution.equals(null));
    }

    @Test
    void testEquals_NotEqual() {
        Assertions.assertFalse(m_theExecution.equals(new Execution(123, 456, "0123456789", "11:11:11", "ACCT123456789", "BATS", "B", 100, 1.0, 1111, 1, 100, 1.0)));
    }

    @Test
    void testEquals_Equal() {
        Assertions.assertTrue(m_theExecution.equals(m_theExecution));
    }
}
