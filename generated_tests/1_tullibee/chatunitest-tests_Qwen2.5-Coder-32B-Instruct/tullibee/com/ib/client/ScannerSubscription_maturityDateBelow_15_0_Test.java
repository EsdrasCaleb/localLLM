package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_15_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateBelow_DefaultValue() {
        // By default, m_maturityDateBelow should be null
        assertNull(scannerSubscription.maturityDateBelow());
    }

    @Test
    public void testMaturityDateBelow_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a value directly to the private field using reflection
        Field maturityDateBelowField = ScannerSubscription.class.getDeclaredField("m_maturityDateBelow");
        maturityDateBelowField.setAccessible(true);
        maturityDateBelowField.set(scannerSubscription, "20231231");
        // Verify that maturityDateBelow() returns the set value
        assertEquals("20231231", scannerSubscription.maturityDateBelow());
    }

    @Test
    public void testMaturityDateBelow_SetValueThroughMethod() {
        // Use the public method to set the value
        scannerSubscription.maturityDateBelow("20240101");
        // Verify that maturityDateBelow() returns the set value
        assertEquals("20240101", scannerSubscription.maturityDateBelow());
    }
}
