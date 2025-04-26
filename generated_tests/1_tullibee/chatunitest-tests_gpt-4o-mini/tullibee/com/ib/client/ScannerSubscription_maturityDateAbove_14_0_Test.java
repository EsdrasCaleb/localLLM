package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_14_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateAbove_WhenNotSet_ShouldReturnNull() {
        // The default value of m_maturityDateAbove should be null
        assertEquals(null, scannerSubscription.maturityDateAbove());
    }

    @Test
    public void testMaturityDateAbove_WhenSet_ShouldReturnCorrectValue() {
        String expectedDate = "2025-12-31";
        scannerSubscription.maturityDateAbove(expectedDate);
        assertEquals(expectedDate, scannerSubscription.maturityDateAbove());
    }

    @Test
    public void testMaturityDateAbove_WhenSetToEmptyString_ShouldReturnEmptyString() {
        String expectedDate = "";
        scannerSubscription.maturityDateAbove(expectedDate);
        assertEquals(expectedDate, scannerSubscription.maturityDateAbove());
    }
}
