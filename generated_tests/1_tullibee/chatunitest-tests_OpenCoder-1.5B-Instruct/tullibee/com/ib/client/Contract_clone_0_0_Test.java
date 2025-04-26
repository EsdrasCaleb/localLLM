package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Contract_clone_0_0_Test {

    @Test
    public void testClone() {
        Contract originalContract = new Contract(12345, "AAPL", "STOCK", "2023-04-01", 150.00, "PE", "1X", "SMART", "USD", "AAPL", new Vector(), "NASDAQ", false, "STK", "AAPL12345");
        try {
            Contract clonedContract = (Contract) originalContract.clone();
            assertEquals(originalContract, clonedContract);
            assertNotSame(originalContract, clonedContract);
        } catch (CloneNotSupportedException e) {
            fail("Clone method should not throw CloneNotSupportedException");
        }
    }
}
