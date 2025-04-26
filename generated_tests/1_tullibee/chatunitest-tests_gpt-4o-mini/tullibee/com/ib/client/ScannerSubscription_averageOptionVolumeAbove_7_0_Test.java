package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_7_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAverageOptionVolumeAbove_DefaultValue() {
        // Test the default value of m_averageOptionVolumeAbove
        assertEquals(Integer.MAX_VALUE, scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_AfterSettingValue() {
        // Set a specific value and test if it retrieves correctly
        int expectedValue = 1000;
        scannerSubscription.averageOptionVolumeAbove(expectedValue);
        assertEquals(expectedValue, scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_AfterSettingToZero() {
        // Set the value to zero and test if it retrieves correctly
        int expectedValue = 0;
        scannerSubscription.averageOptionVolumeAbove(expectedValue);
        assertEquals(expectedValue, scannerSubscription.averageOptionVolumeAbove());
    }
}
