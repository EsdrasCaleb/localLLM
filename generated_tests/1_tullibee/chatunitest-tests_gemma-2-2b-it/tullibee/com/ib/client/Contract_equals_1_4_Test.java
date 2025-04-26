package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Contract_equals_1_4_Test {

    @Test
    void testEquals() {
        Contract contract1 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract2 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract3 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract4 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract5 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract6 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract7 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract8 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract9 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        Contract contract10 = new Contract(1, "ABC", "BOND", "2023-12-31", 100.0, "call", "1.0", "NYSE", "USD", "XYZ", new Vector(), "X", true, "1", "1");
        assertEquals(contract1, contract2);
        assertEquals(contract1, contract3);
        assertEquals(contract1, contract4);
        assertEquals(contract1, contract5);
        assertEquals(contract1, contract6);
        assertEquals(contract1, contract7);
        assertEquals(contract1, contract8);
        assertEquals(contract1, contract9);
        assertEquals(contract1, contract10);
    }
}
