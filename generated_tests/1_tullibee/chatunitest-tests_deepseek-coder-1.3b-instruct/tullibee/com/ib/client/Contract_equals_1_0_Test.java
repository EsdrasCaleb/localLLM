package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Contract_equals_1_0_Test {

    @Test
    void testEquals() {
        Contract contract1 = new Contract();
        Contract contract2 = new Contract();
        assertTrue(contract1.equals(contract2));
        contract1.m_conId = 1;
        assertFalse(contract1.equals(contract2));
        contract2.m_conId = 1;
        assertTrue(contract1.equals(contract2));
        // other fields
        // ...
    }
}
