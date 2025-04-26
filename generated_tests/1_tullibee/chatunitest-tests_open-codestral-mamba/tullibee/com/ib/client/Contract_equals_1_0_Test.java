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
    public void testEquals_SameObject_ReturnsTrue() {
        Contract contract = new Contract();
        assertEquals(contract, contract);
    }

    @Test
    public void testEquals_Null_ReturnsFalse() {
        Contract contract = new Contract();
        assertNotEquals(contract, null);
    }

    @Test
    public void testEquals_DifferentClass_ReturnsFalse() {
        Contract contract = new Contract();
        assertNotEquals(contract, new Object());
    }

    @Test
    public void testEquals_SameValues_ReturnsTrue() {
        Contract contract1 = new Contract(1, "AAPL", "STK", "20220101", 150.0, "BUY", "1", "SMART", "USD", "AAPL", null, "SMART", true, "ISIN", "1");
        Contract contract2 = new Contract(1, "AAPL", "STK", "20220101", 150.0, "BUY", "1", "SMART", "USD", "AAPL", null, "SMART", true, "ISIN", "1");
        assertEquals(contract1, contract2);
    }

    @Test
    public void testEquals_DifferentValues_ReturnsFalse() {
        Contract contract1 = new Contract(1, "AAPL", "STK", "20220101", 150.0, "BUY", "1", "SMART", "USD", "AAPL", null, "SMART", true, "ISIN", "1");
        Contract contract2 = new Contract(2, "MSFT", "STK", "20220101", 200.0, "SELL", "2", "NYSE", "USD", "MSFT", null, "NYSE", true, "CUSIP", "2");
        assertNotEquals(contract1, contract2);
    }
}
