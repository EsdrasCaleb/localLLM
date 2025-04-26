package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_36_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateBelow() throws Exception {
        // Test data
        String testMaturityDate = "20231231";
        // Invoke the method under test
        scannerSubscription.maturityDateBelow(testMaturityDate);
        // Use reflection to access the private field and verify the value
        java.lang.reflect.Field maturityDateBelowField = ScannerSubscription.class.getDeclaredField("m_maturityDateBelow");
        maturityDateBelowField.setAccessible(true);
        String actualMaturityDateBelow = (String) maturityDateBelowField.get(scannerSubscription);
        // Assert that the field has been set correctly
        assertEquals(testMaturityDate, actualMaturityDateBelow, "The maturity date below should be set to the provided value");
    }
}
