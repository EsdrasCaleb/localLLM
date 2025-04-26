package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_40_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testScannerSettingPairs() throws Exception {
        // Given
        String testValue = "testSettingPairs";
        // When
        scannerSubscription.scannerSettingPairs(testValue);
        // Then
        // Using reflection to access the private field
        java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField("m_scannerSettingPairs");
        field.setAccessible(true);
        String result = (String) field.get(scannerSubscription);
        assertEquals(testValue, result, "The scannerSettingPairs field should be set to the provided value.");
    }
}
