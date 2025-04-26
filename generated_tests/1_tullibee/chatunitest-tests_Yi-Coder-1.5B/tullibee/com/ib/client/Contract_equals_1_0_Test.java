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
    void equalsContract() {
        Contract c1 = new Contract(0, "AAPL", "STK", "20210406", 100.0, "CALL", "100", "SMART", "USD", "AAPL", new Vector(), "SMART", false, "STK", "AAPL");
        Contract c2 = new Contract(0, "AAPL", "STK", "20210406", 100.0, "CALL", "100", "SMART", "USD", "AAPL", new Vector(), "SMART", false, "STK", "AAPL");
        Contract c3 = new Contract(1, "AAPL", "STK", "20210406", 100.0, "CALL", "100", "SMART", "USD", "AAPL", new Vector(), "SMART", false, "STK", "AAPL");
        Contract c4 = new Contract(0, "AAPL", "STK", "20210406", 100.0, "CALL", "100", "SMART", "USD", "AAPL", new Vector(), "SMART", false, "STK", "AAPL");
        assertEquals(c1, c2);
        assertNotEquals(c1, c3);
        assertNotEquals(c1, c4);
    }
}
