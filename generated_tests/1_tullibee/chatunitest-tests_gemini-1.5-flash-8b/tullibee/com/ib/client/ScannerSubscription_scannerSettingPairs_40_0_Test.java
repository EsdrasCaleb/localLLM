package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_scannerSettingPairs_40_0_Test {

    @Test
    void testScannerSettingPairs() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid string
        String testString = "someValue";
        subscription.scannerSettingPairs(testString);
        Field scannerSettingPairsField = ScannerSubscription.class.getDeclaredField("m_scannerSettingPairs");
        scannerSettingPairsField.setAccessible(true);
        String actualValue = (String) scannerSettingPairsField.get(subscription);
        assertEquals(testString, actualValue);
        // Test with null
        testString = null;
        subscription.scannerSettingPairs(testString);
        actualValue = (String) scannerSettingPairsField.get(subscription);
        assertEquals(testString, actualValue);
        // Test with an empty string
        testString = "";
        subscription.scannerSettingPairs(testString);
        actualValue = (String) scannerSettingPairsField.get(subscription);
        assertEquals(testString, actualValue);
    }
}
