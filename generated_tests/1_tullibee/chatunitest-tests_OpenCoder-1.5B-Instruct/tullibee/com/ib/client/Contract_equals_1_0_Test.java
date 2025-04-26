package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Contract_equals_1_0_Test {

    @Test
    public void testEquals() {
        // Create instances of Contract
        Contract contract1 = new Contract(1, "AAPL", "STOCK", "2023-12-31", 150.0, "P", "100", "NYSE", "USD", "AAPLUSD", null, "NYSE", false, "STK", "AAPL");
        Contract contract2 = new Contract(1, "AAPL", "STOCK", "2023-12-31", 150.0, "P", "100", "NYSE", "USD", "AAPLUSD", null, "NYSE", false, "STK", "AAPL");
        Contract contract3 = new Contract(2, "AAPL", "STOCK", "2023-12-31", 150.0, "P", "100", "NYSE", "USD", "AAPLUSD", null, "NYSE", false, "STK", "AAPL");
        // Test equal contracts
        assertTrue(contract1.equals(contract2));
        // Test unequal contracts
        assertFalse(contract1.equals(contract3));
    }
}
