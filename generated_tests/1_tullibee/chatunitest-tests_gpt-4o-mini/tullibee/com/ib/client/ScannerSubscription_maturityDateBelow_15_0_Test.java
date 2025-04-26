package com.ib.client;

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
    public void testMaturityDateBelow_WhenNotSet_ShouldReturnNull() {
        // When the maturity date below is not set, it should return null
        assertEquals(null, scannerSubscription.maturityDateBelow());
    }

    @Test
    public void testMaturityDateBelow_WhenSet_ShouldReturnCorrectValue() {
        // Set a value for maturityDateBelow
        String expectedDate = "2025-12-31";
        scannerSubscription.maturityDateBelow(expectedDate);
        // Verify that maturityDateBelow returns the correct value
        assertEquals(expectedDate, scannerSubscription.maturityDateBelow());
    }
}
