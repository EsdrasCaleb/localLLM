package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Contract_clone_0_0_Test {

    @Test
    void testClone() throws CloneNotSupportedException {
        // Test case 1: Normal contract
        Contract contract1 = new Contract(1, "AAPL", "STK", "2024-10-27", 150.0, "CALL", "1", "SMART", "USD", "AAPL", null, "SMART", false, "CUSIP", "12345");
        Vector<String> comboLegs = new Vector<>();
        comboLegs.add("Leg1");
        comboLegs.add("Leg2");
        contract1.m_comboLegs = comboLegs;
        // Mock underComp
        contract1.m_underComp = Mockito.mock(UnderComp.class);
        Contract clonedContract = (Contract) contract1.clone();
        assertNotSame(contract1, clonedContract);
        assertNotEquals(contract1.m_comboLegs, clonedContract.m_comboLegs);
        assertEquals(contract1.m_conId, clonedContract.m_conId);
        assertEquals(contract1.m_symbol, clonedContract.m_symbol);
        assertEquals(contract1.m_secType, clonedContract.m_secType);
        // ... other assertions for other fields ...
        // Check if m_comboLegs vector is cloned
        assertTrue(contract1.m_comboLegs != clonedContract.m_comboLegs);
        assertEquals(contract1.m_comboLegs.size(), clonedContract.m_comboLegs.size());
        // Adding an assertion to verify that the cloned vector contains the same elements as the original.
        for (int i = 0; i < contract1.m_comboLegs.size(); i++) {
            assertEquals(contract1.m_comboLegs.get(i), clonedContract.m_comboLegs.get(i));
        }
        // Test case 2: Empty contract
        Contract contract2 = new Contract();
        Contract clonedContract2 = (Contract) contract2.clone();
        assertNotSame(contract2, clonedContract2);
        // ... other assertions for empty contract ...
    }
}
