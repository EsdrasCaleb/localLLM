package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_19_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testScannerSettingPairs_DefaultValue() {
        // Test the default value of scannerSettingPairs
        assertEquals(null, scannerSubscription.scannerSettingPairs());
    }

    @Test
    public void testScannerSettingPairs_SetValue() {
        // Set a value to scannerSettingPairs and test
        String testValue = "testSettingPairs";
        scannerSubscription.scannerSettingPairs(testValue);
        assertEquals(testValue, scannerSubscription.scannerSettingPairs());
    }
}
