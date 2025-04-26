package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Contract_equals_1_2_Test {

    @Test
    public void testEquals() {
        Contract c1 = new Contract(1, "AAPL", "STOCK", "2022-01-01", 100.0, "C", "USD", "NYSE", "USD", "AAPL", new Vector(), "NYSE", false, "STOCK", "AAPL");
        Contract c2 = new Contract(1, "AAPL", "STOCK", "2022-01-01", 100.0, "C", "USD", "NYSE", "USD", "AAPL", new Vector(), "NYSE", false, "STOCK", "AAPL");
        Contract c3 = new Contract(1, "AAPL", "STOCK", "2022-01-01", 100.0, "C", "USD", "NYSE", "USD", "AAPL", new Vector(), "NYSE", false, "STOCK", "AAPL");
        assertTrue(c1.equals(c2));
        assertTrue(c1.equals(c3));
        assertFalse(c1.equals(null));
        assertFalse(c1.equals(new Object()));
        assertFalse(c1.equals(c1));
    }
}
