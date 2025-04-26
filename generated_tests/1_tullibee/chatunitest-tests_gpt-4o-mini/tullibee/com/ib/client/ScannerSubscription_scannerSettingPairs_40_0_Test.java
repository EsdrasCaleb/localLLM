package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_scannerSettingPairs_40_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    void testScannerSettingPairs_WithValidValue() {
        String testValue = "Test Setting Pair";
        scannerSubscription.scannerSettingPairs(testValue);
        // Using reflection to access the private field
        String actualValue = getPrivateField(scannerSubscription, "m_scannerSettingPairs");
        assertEquals(testValue, actualValue);
    }

    @Test
    void testScannerSettingPairs_WithEmptyValue() {
        String testValue = "";
        scannerSubscription.scannerSettingPairs(testValue);
        // Using reflection to access the private field
        String actualValue = getPrivateField(scannerSubscription, "m_scannerSettingPairs");
        assertEquals(testValue, actualValue);
    }

    @Test
    void testScannerSettingPairs_WithNullValue() {
        String testValue = null;
        scannerSubscription.scannerSettingPairs(testValue);
        // Using reflection to access the private field
        String actualValue = getPrivateField(scannerSubscription, "m_scannerSettingPairs");
        assertEquals(testValue, actualValue);
    }

    private String getPrivateField(ScannerSubscription scannerSubscription, String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (String) field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
