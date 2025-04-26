package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Contract_clone_0_0_Test {

    private Contract contract;

    @BeforeEach
    void setUp() {
        contract = spy(new Contract());
    }

    @Test
    void testClone() throws CloneNotSupportedException {
        Contract clonedContract = (Contract) contract.clone();
        assertNotSame(contract, clonedContract);
        assertEquals(contract.m_conId, clonedContract.m_conId);
        assertEquals(contract.m_symbol, clonedContract.m_symbol);
        // Add assertions for other fields as needed
    }
}
